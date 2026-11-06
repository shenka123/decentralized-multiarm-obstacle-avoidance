"""RRT-Connect algorithm adapter.

Reuses environment.rrt.rrt_connect.birrt directly against a plain
UR5Group -- the same call shape as environment/rrt/rrtWrapper.py's
RRTWrapper.birrt, minus the @ray.remote actor wrapper and the GUI-only
draw_line visualization (this adapter always plans with visualize=False).
"""
from environment.rrt.rrt_connect import birrt

from .base import AlgorithmAdapter

DEFAULT_RESOLUTION = 0.1
DEFAULT_ITERATIONS = 10000
DEFAULT_SMOOTH = 5
DEFAULT_TIMEOUT = 90


class RRTConnectAdapter(AlgorithmAdapter):
    key = 'rrt_connect'
    label = 'RRT-Connect'

    def plan(self, task, ur5_group):
        ur5_group.setup(
            task.base_poses, task.start_config, obs_conf=task.obstacles)
        start_conf = [v for joint_values in task.start_config
                      for v in joint_values]
        goal_conf = [v for joint_values in task.goal_config
                     for v in joint_values]
        path = birrt(
            start_conf=start_conf,
            goal_conf=goal_conf,
            distance=ur5_group.distance_fn,
            sample=ur5_group.sample_fn,
            extend=ur5_group.get_extend_fn(DEFAULT_RESOLUTION),
            collision=ur5_group.get_collision_fn(),
            iterations=DEFAULT_ITERATIONS,
            smooth=DEFAULT_SMOOTH,
            visualize=False,
            fk=ur5_group.forward_kinematics,
            group=True,
            greedy=True,
            timeout=DEFAULT_TIMEOUT,
        )
        if path is None:
            return
        for waypoint in path:
            yield waypoint
