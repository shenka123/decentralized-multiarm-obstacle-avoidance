"""Plans tasks with an OMPL planner or an RL checkpoint, in its own process.

The webapp's own (Windows) env only has pybullet + dash: OMPL ships no
Python bindings on Windows, and the RL rollout pulls in torch/ray/
tensorboardX. So the OMPL and RL adapters (algorithms/ompl.py, rl.py) run
this module in WSL's `multiarm-ompl` conda env instead -- the same env used for
benchmark.py. Running it as a separate process also gives it its own
PyBullet client, so nothing here touches the webapp's single implicit
connection (see sim_manager).

The expensive setup (imports, building the env, loading the checkpoint or
starting ray) is done once, when the worker starts -- i.e. when the
algorithm is picked in the webapp -- and the worker then stays alive,
planning one task per request:

    python -m webapp.plan_worker --checkpoint runs/webapp/obstacle_v5_01191
    python -m webapp.plan_worker --ompl BITstar --timeout 30

Protocol (stdout also carries the env's own prints, so replies are marked):
    worker -> "@@READY"                          once setup is done
    client -> {"task": "tasks/x/0001.json"}      one JSON line per request
    worker -> "@@RESULT " + JSON result          one line per request
The worker exits when stdin closes, e.g. when the webapp quits.

Result JSON:
    frames           [[joint values, flattened across arms], ...]
    obstacle_frames  [{obs_id: [x, y, z]}, ...] (one per frame) or null
    success          whether every arm reached its goal without a collision
    collided         keys ('arm:0', 'obs:s1', ...) of the objects in the
                     collision, [] if none -- see sim_manager.arm_key
    collided_links   [[arm index, link index], ...] of the arm links in
                     contact, [] if none
    collision_frame  index of the frame the collision was detected at, or
                     null; one more frame follows it (see RLPlanner.plan)
    message          one-line summary for the webapp's status bar
"""
import argparse
import json
import sys

import pybullet
import traceback
from time import time

from . import collisions

TORCH_THREADS = 2
DEFAULT_CONFIG = 'configs/evaluate.json'
READY_MARKER = '@@READY'
RESULT_MARKER = '@@RESULT '


class OMPLPlanner:
    def __init__(self, config, planner_name, timeout):
        import ray
        from environment.rrt.omplWrapper import OMPLWrapper

        # OMPLWrapper is a ray actor, so plan exactly as benchmark.py does
        # rather than re-implementing it here.
        self.ray = ray
        ray.init(num_cpus=1, include_dashboard=False, log_to_driver=False)
        self.planner = OMPLWrapper.remote(
            env_config=config['environment'],
            gui=False,
            planner_name=planner_name)
        self.timeout = timeout

    def plan(self, task):
        start = time()
        waypoints = self.ray.get(self.planner.plan_from_task.remote(
            task, timeout=self.timeout))
        elapsed = time() - start
        if not waypoints:
            return {'frames': [], 'obstacle_frames': None, 'success': False,
                    'message': f'No path found ({elapsed:.1f}s).'}
        return {
            'frames': [list(map(float, w)) for w in waypoints],
            'obstacle_frames': None,
            'success': True,
            'message': f'Path found: {len(waypoints)} waypoints '
                       f'in {elapsed:.1f}s.',
        }


class RLPlanner:
    def __init__(self, config, checkpoint):
        import torch
        from environment.benchmarkEnv import BenchmarkEnv
        from policy import StochasticActor

        class SingleTaskEnv(BenchmarkEnv):
            """BenchmarkEnv that plays whichever task plan() sets, with no
            logger or replay memory: episodes are only recorded, never
            scored or trained on."""

            def should_get_next_task(self):
                return False

            def on_episode_end(self):
                pass

        # The policy is tiny: torch's default of one thread per core only
        # adds overhead and starves the webapp (rendering) of CPU.
        torch.set_num_threads(TORCH_THREADS)
        self.torch = torch
        env_config = config['environment']
        training_config = config['training']
        # Any non-None loader stops TaskManager from spawning a TargetEnv
        # actor; it is never queried since should_get_next_task is False.
        training_config['task_loader'] = object()
        self.env = SingleTaskEnv(
            env_config=env_config,
            training_config=training_config,
            gui=False,
            logger=None)
        self.env.set_memory_cluster_map({'multiarm_motion_planner': None})

        self.policy = StochasticActor(
            obs_dim=self.env.actor_observation_dim,
            action_dim=self.env.action_dim,
            action_variance_bounds=training_config['action_variance'],
            network_config=training_config['network']['actor'])
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.policy.load_state_dict(state['networks']['policy'])
        self.policy.eval()

    def act(self, observations):
        from torch.nn.utils.rnn import pad_sequence

        obs = pad_sequence(
            observations['multiarm_motion_planner'], batch_first=True)
        with self.torch.no_grad():
            actions, _ = self.policy(
                obs, deterministic=True, reparametrize=False)
        return actions.cpu()

    def step_past_end(self, actions):
        """One control step of physics after the episode has ended -- the
        same as BaseEnv.step's inner loop, minus rewards and termination
        (env.step itself would just reset the episode)."""
        env = self.env
        env.action_to_robots(actions)
        for _ in range(env.simulation_steps_per_action_step):
            env.sim_time += env.SIMULATION_TIMESTEP
            env.update_obstacles()
            pybullet.stepSimulation()
            for ur5 in env.active_ur5s:
                ur5.step()

    def snapshot(self):
        joints = [v for ur5 in self.env.active_ur5s
                  for v in ur5.get_arm_joint_values()]
        obstacles = {obs_id: list(obs.get_position())
                     for obs_id, obs in self.env.active_obs.items()}
        return [float(v) for v in joints], obstacles

    def plan(self, task):
        env = self.env
        env.task_manager.current_task = task
        start = time()
        observations, _ = env.reset()
        frames, obstacle_frames = [], []
        joints, obstacles = self.snapshot()
        frames.append(joints)
        obstacle_frames.append(obstacles)
        while not env.terminate_episode:
            observations, _ = env.step(self.act(observations))
            joints, obstacles = self.snapshot()
            frames.append(joints)
            obstacle_frames.append(obstacles)

        collided = sum(env.stats['collisions']) > 0
        reached = env.stats['collective_reach_count'] > 0
        steps = len(frames) - 1
        # Read the contacts before the extra step below: physics pushes the
        # bodies apart during it.
        contacts = self._contacts() if collided else []
        if collided:
            # The episode ends at the first contact, where the arms barely
            # touch, so the last frame hardly shows it. Play one more
            # control step so the replay ends visibly in the collision.
            self.step_past_end(self.act(observations))
            joints, obstacles = self.snapshot()
            frames.append(joints)
            obstacle_frames.append(obstacles)
        elapsed = time() - start
        if collided:
            outcome = (f'Collision ({collisions.describe(contacts)}) '
                       f'at step {steps}')
        elif reached:
            outcome = f'All arms reached their goals at step {steps}'
        else:
            if env.current_idle_step >= env.early_stop_steps:
                outcome = (f'Stopped early at step {steps}: arms barely '
                           f'moved for {env.early_stop_steps} steps')
            else:
                outcome = f'Timed out after {steps} steps'
        return {
            'frames': frames,
            # Only worth replaying when something actually moves.
            'obstacle_frames': (obstacle_frames if task.obstacle_motion
                                else None),
            'success': reached and not collided,
            'collided': collisions.collided_objects(contacts),
            'collided_links': collisions.collided_links(contacts),
            'collision_frame': steps if collided else None,
            'message': f'{outcome} ({elapsed:.1f}s).',
        }

    def _contacts(self):
        """Contacts of the arms colliding at the last step (see
        collisions.find_contacts)."""
        env = self.env
        colliding = [i for i, state in enumerate(env.state['ur5s'])
                     if state['colliding']]
        return collisions.find_contacts(
            env.active_ur5s, env.active_obs, colliding,
            env.collision_distance)


def reply(marker, payload=None):
    line = marker if payload is None else marker + json.dumps(payload)
    sys.stdout.write(line + '\n')
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    method = parser.add_mutually_exclusive_group(required=True)
    method.add_argument('--ompl', metavar='PLANNER')
    method.add_argument('--checkpoint')
    parser.add_argument('--timeout', type=float, default=30.0)
    # Override the config's environment.early_stop (RL only).
    parser.add_argument('--early-stop-threshold', type=float)
    parser.add_argument('--early-stop-steps', type=int)
    args = parser.parse_args()

    from environment.tasks import Task
    from utils import load_config

    config = load_config(args.config)
    early_stop = config['environment']['early_stop']
    if args.early_stop_threshold is not None:
        early_stop['threshold'] = args.early_stop_threshold
    if args.early_stop_steps is not None:
        early_stop['steps'] = args.early_stop_steps
    if args.ompl:
        planner = OMPLPlanner(config, args.ompl, args.timeout)
    else:
        planner = RLPlanner(config, args.checkpoint)
    reply(READY_MARKER)

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            task = Task.from_file(json.loads(line)['task'])
            result = planner.plan(task)
        except Exception as exc:
            traceback.print_exc()
            result = {'error': f'{type(exc).__name__}: {exc}'}
        reply(RESULT_MARKER, result)


if __name__ == '__main__':
    main()
