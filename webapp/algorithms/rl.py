"""Trained RL policy adapter: rolls a checkpoint out in WSL.

The checkpoint is loaded once, in prepare() (i.e. when the algorithm is
picked), by a plan_worker process that then stays alive for every run.

The rollout runs the real BenchmarkEnv (see plan_worker.RLPlanner), so the
frames are what the policy actually did, collisions and moving obstacles
included -- unlike the OMPL adapter, a failed episode still has frames to
replay, ending at the step it failed.
"""
from .. import app_config
from .base import AlgorithmAdapter
from .wsl_worker import WorkerProcess, to_wsl_path

# One episode is at most BenchmarkEnv's 20 s of simulated time; the
# rollout itself takes seconds, so this only guards against a hung worker.
ROLLOUT_TIMEOUT = 300


class RLAdapter(AlgorithmAdapter):
    uses_sim = False

    def __init__(self, checkpoint_path):
        super().__init__()
        config = app_config.load()
        args = ['--checkpoint', to_wsl_path(checkpoint_path),
                '--config', config.get('rl_config',
                                       app_config.DEFAULT_RL_CONFIG)]
        early_stop = config.get('early_stop') or {}
        if 'threshold' in early_stop:
            args += ['--early-stop-threshold', str(early_stop['threshold'])]
        if 'steps' in early_stop:
            args += ['--early-stop-steps', str(early_stop['steps'])]
        self.worker = WorkerProcess(*args)

    def prepare(self):
        self.worker.start()

    def close(self):
        self.worker.close()

    def plan(self, task, ur5_group):
        result = self.worker.plan(task, timeout=ROLLOUT_TIMEOUT)
        self.obstacle_frames = result['obstacle_frames']
        self.success = result['success']
        self.collided = result.get('collided', [])
        self.collided_links = [tuple(link) for link in
                               result.get('collided_links', [])]
        self.collision_frame = result.get('collision_frame')
        self.result_message = result['message']
        yield from result['frames']
