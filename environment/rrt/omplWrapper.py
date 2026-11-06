from itertools import chain
import ray
import pybullet as p
import pybullet_data
from .ur5_group import UR5Group
from environment.UR5 import UR5
from ompl import base as ob
from ompl import geometric as og


@ray.remote
class OMPLWrapper:
    def __init__(self, env_config, gui=False, planner_name='RRTConnect'):
        from environment.utils import create_ur5s, create_obstacles
        from environment.rrt.pybullet_utils import configure_pybullet

        configure_pybullet(
            rendering=gui,
            debug=False,
            yaw=150,
            pitch=-30,
            dist=2.,
            target=(0, 0, 0.1))

        self.gui = gui
        self.planner_name = planner_name

        p.loadURDF(
            "plane.urdf",
            [0, 0, -env_config['collision_distance'] - 0.01])

        self.ur5_group = UR5Group(
            create_ur5s_fn=lambda: create_ur5s(
                radius=0.8,
                count=env_config['max_ur5s_count'],
                speed=env_config['ur5_speed'],
                ee_hitbox=env_config.get('ee_hitbox', 'training')),
            collision_distance=env_config['collision_distance'],
            all_obs=create_obstacles(env_config['obstacles']))

    def plan_from_task(self, task, timeout=30.0):
        """
        Takes a Task object, returns List[List[float]] waypoints
        in the same flat joint config format as RRTWrapper.birrt_from_task,
        or None if planning fails.
        """
        

        self.ur5_group.setup(
            start_poses=task.base_poses,
            start_joints=task.start_config,
            obs_conf=task.obstacles)

        collision_fn = self.ur5_group.get_collision_fn()
        dof = self.ur5_group.compute_dof()

        flat_start = list(chain.from_iterable(task.start_config))
        goal_config = task.start_goal_config
        if goal_config is None or any(g is None for g in goal_config):
            goal_config = task.goal_config
        flat_goal = list(chain.from_iterable(goal_config))

        if collision_fn(flat_start):
            print(f"[OMPLWrapper] Start config in collision for task {task.id}")
            return None
        if collision_fn(flat_goal):
            print(f"[OMPLWrapper] Goal config in collision for task {task.id}")
            return None

        return self._ompl_plan(flat_start, flat_goal,
                               collision_fn, dof, timeout)

    def _ompl_plan(self, flat_start, flat_goal,
                   collision_fn, dof, timeout):
        # Build state space using UR5 joint limits replicated per arm
        space = ob.RealVectorStateSpace(dof)
        bounds = ob.RealVectorBounds(dof)

        n_arms = dof // 6
        flat_lower = list(UR5.LOWER_LIMITS) * n_arms
        flat_upper = list(UR5.UPPER_LIMITS) * n_arms

        for i, (lo, hi) in enumerate(zip(flat_lower, flat_upper)):
            bounds.setLow(i, lo)
            bounds.setHigh(i, hi)
        space.setBounds(bounds)

        si = ob.SpaceInformation(space)
        class ValidityChecker(ob.StateValidityChecker):
            def __init__(self, si, collision_fn, dof):
                super().__init__(si)
                self._collision_fn = collision_fn
                self._dof = dof

            def isValid(self, state):
                q = [state[i] for i in range(self._dof)]
                return not self._collision_fn(q)

        validity_checker = ValidityChecker(si, collision_fn, dof)
        si.setStateValidityChecker(validity_checker)
        si.setup()

        pdef = ob.ProblemDefinition(si)    
        start_state = si.allocState()
        goal_state = si.allocState()
        for i, v in enumerate(flat_start):
            start_state[i] = v
        for i, v in enumerate(flat_goal):
            goal_state[i] = v
        pdef.setStartAndGoalStates(start_state, goal_state)

        planner = self._make_planner(si)
        planner.setProblemDefinition(pdef)
        planner.setup()

        solved = planner.solve(timeout)
        if not solved:
            return None

        path = pdef.getSolutionPath()

        # Simplify then interpolate for smooth execution
        path_simplifier = og.PathSimplifier(si)
        try:
            path_simplifier.simplify(path, timeout / 2)
        except Exception:
            pass  # simplification is best-effort

        path.interpolate(200)

        return [
            [path.getState(i)[j] for j in range(dof)]
            for i in range(path.getStateCount())
        ]

    def _make_planner(self, si):
        planner_map = {
            'RRTConnect': og.RRTConnect,
            'RRT':        og.RRT,
            'BITstar':    og.BITstar,
            'PRM':        og.PRM,
            'LBKPIECE':   og.LBKPIECE1,
        }
        cls = planner_map.get(self.planner_name, og.RRTConnect)
        return cls(si)