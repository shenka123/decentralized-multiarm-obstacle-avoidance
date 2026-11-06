"""OMPL planner adapter (BIT*, RRT-Connect, ...), planned in WSL.

Same planner and settings as benchmark.py's OMPL baselines: it goes through
environment.rrt.omplWrapper.OMPLWrapper inside plan_worker, whose ray
instance is started once in prepare() and reused for every run.

OMPLWrapper always returns its path interpolated to 200 states, however
long the path is, so replayed as-is every OMPL run lasts 200 frames and a
short motion crawls. retime() resamples it instead into frames of one RL
control step (0.05 s) each, so OMPL and RL replays move at comparable
speeds: no joint moves more than max_joint_step per frame, and with
max_joint_accel set the speed ramps up from rest, cruises and ramps back
down (a trapezoidal profile) rather than starting and stopping instantly.
Both limits come from "ompl_playback" in webapp/config.json (see
app_config). Only the replay changes -- the planned path is the same.

OMPL plans against the obstacles where the task starts them. With moving
obstacles, the replay still moves them along their motion, one control
step per frame as in an RL rollout, while the arms follow that fixed
path -- so they may collide. check_replay() finds the first frame in
collision, with the same check as the RL environment's, flags it (red
links and obstacles) and ends the replay there, as an RL episode ends.
"""
import math

import numpy as np

from environment.obstacle_motion import ObstacleMotion

from .. import app_config, collisions
from .base import AlgorithmAdapter
from .wsl_worker import WorkerProcess

DEFAULT_TIMEOUT = 10  # per-task planning budget, in seconds
FRAME_SECONDS = 0.05  # one RL control step (action_interval)

# Per RL control step (0.05 s), in rad. Measured from the Baseline
# checkpoint on tasks/_obstacle_evaluate: the fastest joint's step peaks
# at ~0.06 per episode (median), and its step-to-step change mostly stays
# within 0.005-0.02.
DEFAULT_MAX_JOINT_STEP = 0.06
DEFAULT_MAX_JOINT_ACCEL = 0.01


def retime(waypoints, max_step, max_accel=None):
    """Resamples `waypoints` (joint configs, flattened across arms) along
    the same piecewise-linear path into frames where no joint moves more
    than `max_step` per frame. With `max_accel` (the most that per-frame
    step may change from one frame to the next) the speed follows a
    trapezoidal profile from rest to rest; with None it is constant.
    The first and last frames are the path's own start and goal."""
    q = np.asarray(waypoints, dtype=float)
    if len(q) < 2:
        return q.tolist()
    # Arc length in the max-norm: moving ds along a segment moves every
    # joint by at most ds, so capping ds per frame caps every joint.
    seg = np.abs(np.diff(q, axis=0)).max(axis=1)
    keep = np.concatenate([[True], seg > 1e-12])
    q, seg = q[keep], seg[seg > 1e-12]
    if len(q) < 2:
        return q.tolist()
    s = np.concatenate([[0.0], np.cumsum(seg)])
    length = s[-1]

    if max_accel:
        # Peak speed: max_step, or lower if the path is too short to reach
        # it (then the profile is a triangle).
        peak = min(max_step, math.sqrt(length * max_accel))
        ramp = peak / max_accel  # frames to reach the peak
        duration = 2 * ramp + (length - peak * ramp) / peak
    else:
        peak, ramp = max_step, 0.0
        duration = length / max_step
    # Whole frames; stretching to land exactly on the goal only slows the
    # profile down, so both limits still hold.
    frames = max(1, math.ceil(duration - 1e-9))
    t = np.arange(frames + 1) * (duration / frames)

    if ramp:
        cruise_end = duration - ramp
        dist = np.where(
            t < ramp, 0.5 * max_accel * t ** 2,
            np.where(t <= cruise_end,
                     0.5 * peak * ramp + peak * (t - ramp),
                     length - 0.5 * max_accel * (duration - t) ** 2))
    else:
        dist = peak * t
    dist = np.clip(dist, 0.0, length)
    dist[-1] = length

    out = np.stack([np.interp(dist, s, q[:, j])
                    for j in range(q.shape[1])], axis=1)
    return out.tolist()


def playback_limits():
    """(max_joint_step, max_joint_accel or None) from config.json's
    ompl_playback, or None to replay OMPL's waypoints unchanged."""
    playback = app_config.load().get('ompl_playback', {})
    if not isinstance(playback, dict):
        playback = {}
    if playback.get('enabled', True) is False:
        return None
    step = playback.get('max_joint_step', DEFAULT_MAX_JOINT_STEP)
    accel = playback.get('max_joint_accel', DEFAULT_MAX_JOINT_ACCEL)
    return float(step), (float(accel) if accel else None)


def obstacle_frames(task, frame_count):
    """[{obs_id: position}, ...], one per frame, of the task's moving
    obstacles at FRAME_SECONDS per frame from t = 0, or None if none move.
    Same closed-form motion BaseEnv plays them along."""
    motions = {}
    for obs_id, origin in task.obstacles.items():
        motion = ObstacleMotion.from_task(task.obstacle_motion, obs_id, origin)
        if motion is not None:
            motions[obs_id] = motion
    if not motions:
        return None
    t = np.arange(frame_count) * FRAME_SECONDS
    paths = {obs_id: motion.position_at(t).tolist()
             for obs_id, motion in motions.items()}
    return [{obs_id: path[k] for obs_id, path in paths.items()}
            for k in range(frame_count)]


class OMPLAdapter(AlgorithmAdapter):
    uses_sim = False

    def __init__(self, planner_name, timeout=DEFAULT_TIMEOUT):
        super().__init__()
        self.timeout = timeout
        self.worker = WorkerProcess(
            '--ompl', planner_name, '--timeout', str(timeout))

    def prepare(self):
        self.worker.start()

    def close(self):
        self.worker.close()

    def plan(self, task, ur5_group):
        # BIT* can spend up to `timeout` solving + timeout/2 simplifying.
        result = self.worker.plan(task, timeout=self.timeout * 1.5 + 30)
        self.collided, self.collided_links = [], []  # set by check_replay
        self.collision_frame = None
        self.success = result['success']
        self.result_message = result['message']
        frames = result['frames']
        limits = playback_limits()
        if frames and limits:
            frames = retime(frames, *limits)
            self.result_message += (
                f' Retimed to {len(frames)} frames '
                f'({(len(frames) - 1) * FRAME_SECONDS:.1f} s).')
        self.obstacle_frames = (obstacle_frames(task, len(frames))
                                if frames else None)
        yield from frames

    def check_replay(self, task, frames, ur5_group):
        """Poses every frame (obstacles included) and cuts the replay at
        the first one where an arm collides, checked as BaseEnv does
        (UR5.check_collision at the scene's collision distance)."""
        arms = ur5_group.active_controllers
        margin = ur5_group.collision_distance
        for k, frame in enumerate(frames):
            if self.obstacle_frames:
                for obs_id, pos in self.obstacle_frames[k].items():
                    ur5_group.active_obs[obs_id].set_position(pos)
            ur5_group.set_joint_positions(frame)
            colliding = [i for i, ur5 in enumerate(arms)
                         if ur5.check_collision(collision_distance=margin)]
            if colliding:
                contacts = collisions.find_contacts(
                    arms, ur5_group.active_obs, colliding, margin)
                self.collided = collisions.collided_objects(contacts)
                self.collided_links = [tuple(link) for link in
                                       collisions.collided_links(contacts)]
                self.collision_frame = k
                self.success = False
                self.result_message += (
                    f' Collision ({collisions.describe(contacts)}) '
                    f'at frame {k}; replay ends there.')
                del frames[k + 1:]
                if self.obstacle_frames:
                    del self.obstacle_frames[k + 1:]
                return
