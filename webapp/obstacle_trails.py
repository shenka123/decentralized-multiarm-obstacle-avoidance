"""Fading "ghost" trail of moving obstacles, part of Show trajectories.

Only drawn when obstacles move (SimulationManager.obstacle_frames). Each
ghost is a visual-only copy of an obstacle at an earlier frame: one every
GHOST_STRIDE frames, the newest GHOST_COUNT of them, fading from
NEWEST_ALPHA to OLDEST_ALPHA with age, so the trail shows where the
obstacle came from and how recently.

Like path_markers, sync() diffs against the ghosts already in the scene, so
a playback tick adds/removes at most one ghost per obstacle; the alphas of
the rest are updated in place, since every ghost ages by one frame per tick.
"""
import pybullet as p

GHOST_STRIDE = 32  # 1.6 s at the 0.05 s control step; obstacles move slowly
GHOST_COUNT = 5  # trail covers the last 8 s
NEWEST_ALPHA = 0.6
OLDEST_ALPHA = 0.15


def desired_frame_indices(frame_index):
    """Past frames on the stride grid, newest first, excluding the current
    frame (where the obstacle itself is)."""
    if frame_index <= 0:
        return []
    newest = (frame_index - 1) // GHOST_STRIDE * GHOST_STRIDE
    return list(range(newest, -1, -GHOST_STRIDE))[:GHOST_COUNT]


def ghost_alpha(age_rank):
    """age_rank 0 is the newest ghost."""
    if GHOST_COUNT == 1:
        return NEWEST_ALPHA
    return NEWEST_ALPHA + (OLDEST_ALPHA - NEWEST_ALPHA) * (
        age_rank / (GHOST_COUNT - 1))


class ObstacleTrails:
    def __init__(self):
        self._bodies = {}  # (frame_index, obs_id) -> body id
        self._shapes = {}  # obs_id -> visual shape, reused by its ghosts

    def sync(self, obstacle_frames, frame_index, obstacles, shown):
        """obstacle_frames: per-frame {obs_id: pos} (or None to clear);
        obstacles: {obs_id: Obstacle}; shown(obs_id): whether that
        obstacle's ghosts are drawn (e.g. only highlighted ones)."""
        indices = (desired_frame_indices(frame_index)
                   if obstacle_frames else [])
        desired = {(i, obs_id) for i in indices
                   for obs_id in obstacle_frames[i]
                   if obs_id in obstacles and shown(obs_id)}
        for key in set(self._bodies) - desired:
            p.removeBody(self._bodies.pop(key))
        age_rank = {i: rank for rank, i in enumerate(indices)}
        for key in desired:
            i, obs_id = key
            obstacle = obstacles[obs_id]
            if key not in self._bodies:
                self._bodies[key] = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=self._shape(obs_id, obstacle),
                    basePosition=obstacle_frames[i][obs_id])
            p.changeVisualShape(
                self._bodies[key], -1,
                rgbaColor=list(obstacle.color[:3]) + [ghost_alpha(age_rank[i])])

    def clear(self):
        for body_id in self._bodies.values():
            p.removeBody(body_id)
        self._bodies = {}

    def _shape(self, obs_id, obstacle):
        if obs_id not in self._shapes:
            self._shapes[obs_id] = p.createVisualShape(
                p.GEOM_SPHERE, radius=obstacle.radius,
                rgbaColor=list(obstacle.color[:3]) + [NEWEST_ALPHA])
        return self._shapes[obs_id]
