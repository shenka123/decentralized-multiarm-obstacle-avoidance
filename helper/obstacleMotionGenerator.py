"""
obstacleMotionGenerator.py

Adds a movement seed ("obstacle_motion") to tasks that already have obstacles.
The obstacle paths are simulated over the evaluation horizon, and a seed is
rejected if any obstacle collides with an arm placed at its goal config
(this covers the arm base and the end effector at its goal pose), or if any
obstacle's center goes below the floor (z = 0). Expert
trajectories are ignored. Rejected seeds are resampled up to max_attempts (50).

Tasks with an accepted seed are written to tasks/{target_name}/.
Tasks that never get a collision-free seed are logged as failed and skipped.

Usage:
    python -m helper.obstacleMotionGenerator <task_name> <target_name> <num_workers> [reset]

    python -m helper.obstacleMotionGenerator obstacle_evaluate obstacle_evaluate_moving 10
    python -m helper.obstacleMotionGenerator obstacle_evaluate obstacle_evaluate_moving 10 reset
"""

import ray
import numpy as np
from environment.rrt.pybullet_utils import configure_pybullet
from environment.rrt.ur5_group import UR5Group
from environment.obstacle_motion import ObstacleMotion, DEFAULT_MOTION
from helper.obstacleGenerator import iter_tasks, dump_json
import json
import sys
import os


def build_motion(motion_params):
    """Sample a new movement seed with the given motion parameters."""
    return {
        "movement_seed": int(np.random.randint(0, 2**31 - 1)),
        **motion_params,
    }


def get_goal_config(task_data):
    goal_config = task_data.get('start_goal_config')
    if goal_config is None or any(g is None for g in goal_config):
        goal_config = task_data['goal_config']
    return goal_config


# ---------------------------------------------------------------------------
# Motion validator actor
# ---------------------------------------------------------------------------

@ray.remote
class MotionValidator:
    """
    Ray actor: one PyBullet instance that sweeps obstacles along their
    seeded paths and checks them against the arms at goal config.
    """

    def __init__(self, env_config, gui=False):
        from environment.utils import create_ur5s, create_obstacles

        configure_pybullet(
            rendering=gui,
            debug=False,
            yaw=150, pitch=-30,
            dist=2., target=(0, 0, 0.1))

        import pybullet as p
        import pybullet_data
        p.loadURDF("plane.urdf",
                   [0, 0, -env_config['collision_distance'] - 0.01])

        self.collision_distance = env_config['collision_distance']
        self.ur5_group = UR5Group(
            create_ur5s_fn=lambda: create_ur5s(
                radius=0.8,
                count=env_config['max_ur5s_count'],
                speed=env_config['ur5_speed']),
            collision_distance=env_config['collision_distance'],
            all_obs=create_obstacles(env_config['obstacles']))

    def validate(self, task_data, obstacle_motion, duration, check_dt):
        """
        Returns True if no obstacle touches any arm (at goal config) and no
        obstacle center is below the floor at any sampled t in [0, duration].
        """
        import pybullet as p

        obstacles = task_data.get('obstacles', {})
        if len(obstacles) == 0:
            return True

        self.ur5_group.setup(
            start_poses=task_data['base_poses'],
            start_joints=get_goal_config(task_data),
            obs_conf=obstacles)

        ts = np.arange(0.0, duration + check_dt, check_dt)
        paths = {
            obs_id: ObstacleMotion.from_task(
                obstacle_motion, obs_id, obstacles[obs_id]).position_at(ts)
            for obs_id in self.ur5_group.active_obs
        }

        # Reject paths where a sphere's center goes below the floor (z = 0)
        for obs_id in self.ur5_group.active_obs:
            if paths[obs_id][:, 2].min() < 0:
                return False

        for i in range(len(ts)):
            for obs_id, obs in self.ur5_group.active_obs.items():
                obs.set_position(paths[obs_id][i])
                for c in self.ur5_group.active_controllers:
                    if len(p.getClosestPoints(
                            bodyA=c.body_id, bodyB=obs.body_id,
                            distance=self.collision_distance)) > 0:
                        return False
        return True


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def generate_obstacle_motion(
        task_name='',
        target_name='',
        config_file='configs/RRTconfig.json',
        motion_params=DEFAULT_MOTION,
        duration=None,
        check_dt=0.05,
        reset=False,
        max_attempts=50,
        num_workers=10,
        gui=False):
    """
    Add collision-free movement seeds to tasks with obstacles.

    tasks/{task_name}/     — source tasks (with "obstacles")
    tasks/{target_name}/   — output tasks with "obstacle_motion" added
    logs/motion_{target_name}.json — progress log

    duration: simulated seconds to check; defaults to
              episode_length * action_interval of config_file.
    """

    print("Initialising Ray...")
    ray.init()

    with open(config_file) as f:
        env_config = json.load(f)['environment']
    if duration is None:
        duration = env_config['episode_length'] * env_config['action_interval']
    print(f"Checking obstacle paths over {duration:.2f}s "
          f"every {check_dt}s with {motion_params}")

    tasks_dir  = f'tasks/{task_name}/'
    target_dir = f'tasks/{target_name}/'
    logs_file  = f'logs/motion_{target_name}.json'

    logs = {
        "stats": {"processed": 0, "success": 0, "failed": 0, "errors": 0},
        "runs": {}
    }
    if not reset:
        try:
            logs = json.load(open(logs_file))
        except Exception:
            pass

    for d in [target_dir, 'logs/', 'temp/']:
        os.makedirs(d, exist_ok=True)

    # ── Worker pool ────────────────────────────────────────────────────────
    print(f"Creating {num_workers} MotionValidator workers...")
    workers = [
        MotionValidator.remote(env_config=env_config, gui=gui)
        for _ in range(num_workers)
    ]

    # ── Task queue ─────────────────────────────────────────────────────────
    pending = iter_tasks(tasks_dir, logs)

    # worker_state: {worker: {future, filename, task_data, attempt, motion}}
    worker_state = {}
    done_count = 0

    def submit(worker, state):
        state["motion"] = build_motion(motion_params)
        state["future"] = worker.validate.remote(
            state["task_data"], state["motion"], duration, check_dt)

    def assign_next_task(worker):
        try:
            filename, task_data, task_file = next(pending)
        except StopIteration:
            return False
        worker_state[worker] = {
            "filename":  filename,
            "task_data": task_data,
            "attempt":   1,
        }
        submit(worker, worker_state[worker])
        return True

    def finish(worker, result, attempt):
        nonlocal done_count
        logs["runs"][worker_state[worker]["filename"]] = f"{result}:{attempt}"
        logs["stats"]["processed"] += 1
        logs["stats"]["success" if result == "success" else
                      "failed" if result == "failed" else "errors"] += 1
        done_count += 1
        del worker_state[worker]
        assign_next_task(worker)

    # Seed workers
    for worker in workers:
        if not assign_next_task(worker):
            break

    # ── Main loop ──────────────────────────────────────────────────────────
    while worker_state:
        if done_count % 100 == 0:
            dump_json(logs, logs_file, save_write=True)

        all_futures      = [s["future"] for s in worker_state.values()]
        future_to_worker = {s["future"]: w for w, s in worker_state.items()}

        ready, _ = ray.wait(all_futures, num_returns=1, timeout=60.0)

        if not ready:
            future = all_futures[0]
            worker = future_to_worker[future]
            ray.cancel(future, force=True)
            valid = False
        else:
            future = ready[0]
            worker = future_to_worker[future]
            try:
                valid = ray.get(future, timeout=5)
            except Exception as e:
                print(f"[ERROR] Ray get failed: {e}")
                finish(worker, "error", 0)
                continue

        state   = worker_state[worker]
        attempt = state["attempt"]

        if valid:
            task_data = state["task_data"]
            task_data['obstacle_motion'] = state["motion"]
            dump_json(task_data, target_dir + state["filename"])
            finish(worker, "success", attempt)
        elif attempt < max_attempts:
            state["attempt"] += 1
            submit(worker, state)
        else:
            finish(worker, "failed", attempt)

    dump_json(logs, logs_file, save_write=True)

    s = logs["stats"]
    print(f"\nDone!  processed={s['processed']}  "
          f"success={s['success']}  failed={s['failed']}  "
          f"errors={s['errors']}")
    ray.shutdown()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    generate_obstacle_motion(
        task_name=sys.argv[1],
        target_name=sys.argv[2],
        num_workers=int(sys.argv[3]),
        reset="reset" in sys.argv,
        config_file='configs/RRTconfig.json',
        gui=False,
    )
