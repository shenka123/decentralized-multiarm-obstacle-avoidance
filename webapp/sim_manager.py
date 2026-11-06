"""Owns the single PyBullet client used by the whole webapp.

This is a local, single-operator tool (not a multi-tenant service), so one
global SimulationManager instance backs the entire app -- see the plan's
"single global simulation" scoping decision. Every PyBullet call anywhere
in the webapp must go through this instance's lock, since PyBullet itself
isn't thread-safe across concurrent calls.

Connects via p.GUI (not p.DIRECT) and immediately hides the window: this
gives getCameraImage a real OpenGL context (see render.py) instead of
falling back to TinyRenderer's slow software rasterization, which is what
made playback feel sluggish -- rendering, not frame-advancement, was the
bottleneck. The window flashes briefly on startup only; _hide_own_windows
is Windows-specific (this is a Windows-only deployment for now).
"""
import ctypes
import json
import math
import os
import threading
import time
from pathlib import Path

import pybullet as p
import pybullet_data

from environment.rrt.ur5_group import UR5Group
from environment.utils import create_ur5s, create_obstacles, Target

from . import app_config, path_markers, render
from .exporter import Exporter
from .obstacle_trails import ObstacleTrails
from .algorithms.registry import create_adapter, list_adapters

REPO_ROOT = Path(__file__).resolve().parent.parent
# Base obstacle set (ids -> type/radius) that every task's `obstacles` dict
# (ids -> position) is validated against. obstacle_v1.json's 5 spheres
# (s1..s5) match the obstacle ids used throughout tasks/base_obs*.
OBSTACLE_CONFIG_PATH = REPO_ROOT / 'configs' / 'obstacle_v1.json'
MAX_UR5S_COUNT = 4
UR5_SPEED = 0.02
COLLISION_DISTANCE = 0.0
PLAYBACK_FPS = 10  # at 1x speed
PLAYBACK_SPEEDS = (1.0, 2.0, 0.5)  # cycled through by the speed button
STEP_SIZE = 10  # frames moved by one rewind/skip click
MAX_LOG_ENTRIES = 500
HIGHLIGHT_DIM = 0.2  # opacity multiplier for objects that aren't highlighted
COLLISION_RGB = [0.9, 0.1, 0.1]  # links/obstacles in contact, from the
                                 # collision frame on
REACHED_RGB = [0.4, 1.0, 0.4]  # as Target.touched(): goal markers and end
                               # effectors at their goals
DEFAULT_GOAL_TOLERANCES = (0.05, 0.2)  # m, rad: if rl_config has none


# Every arm/obstacle in the scene is referred to by a key -- shared by the
# highlight checkboxes, the labels and the collision info from plan_worker.
def arm_key(index):
    return f'arm:{index}'


def obstacle_key(obs_id):
    return f'obs:{obs_id}'


def object_label(key):
    kind, name = key.split(':', 1)
    return f'Arm {int(name) + 1}' if kind == 'arm' else f'Obstacle {name}'


def _load_base_obstacles_config():
    with open(OBSTACLE_CONFIG_PATH) as f:
        config = json.load(f)
    return config['environment']['obstacles']


def goal_tolerances():
    """(position, orientation) tolerance of reaching a goal, from the
    environment config the RL planners run with (webapp/config.json's
    rl_config), so green means what success means to the policy."""
    path = app_config.load().get('rl_config', app_config.DEFAULT_RL_CONFIG)
    try:
        with open(REPO_ROOT / path) as f:
            env_config = json.load(f)['environment']
        return (float(env_config['position_tolerance']),
                float(env_config['orientation_tolerance']))
    except (OSError, ValueError, KeyError) as exc:
        print(f'[sim_manager] Goal tolerances from {path}: {exc}; '
              f'using the defaults')
        return DEFAULT_GOAL_TOLERANCES


def _hide_own_windows():
    """Hide every top-level window belonging to this process (Windows only)."""
    if os.name != 'nt':
        return []
    user32 = ctypes.windll.user32
    pid = os.getpid()
    hidden = []
    enum_windows_proc = ctypes.WINFUNCTYPE(
        ctypes.c_bool, ctypes.c_int, ctypes.c_int)

    def callback(hwnd, _lparam):
        found_pid = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(found_pid))
        if found_pid.value == pid and user32.IsWindowVisible(hwnd):
            user32.ShowWindow(hwnd, 0)  # SW_HIDE
            hidden.append(hwnd)
        return True

    user32.EnumWindows(enum_windows_proc(callback), 0)
    return hidden


class _SceneLock:
    """RLock that counts how often it's taken: every change to the scene
    happens under it, so an unchanged count means an unchanged scene and
    render_jpeg can reuse its last image."""

    def __init__(self):
        self._lock = threading.RLock()
        self.version = 0

    def __enter__(self):
        self._lock.acquire()
        self.version += 1
        return self

    def __exit__(self, *exc):
        self._lock.release()


class SimulationManager:
    def __init__(self):
        self.lock = _SceneLock()
        self._last_render = (None, None)  # (key, jpeg bytes)
        self.exporter = Exporter(self)
        with self.lock:
            self.client_id = p.connect(p.GUI)
            _hide_own_windows()
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.resetSimulation()
            p.setGravity(0, 0, -9.81)
            # Robot bases sit at z=0 -- offset the plane down so they don't
            # spuriously register as colliding with it (matches
            # environment/baseEnv.py's convention).
            p.loadURDF('plane.urdf', [0, 0, -COLLISION_DISTANCE - 0.01])
            self.ur5_group = UR5Group(
                create_ur5s_fn=lambda: create_ur5s(
                    radius=0.8, count=MAX_UR5S_COUNT, speed=UR5_SPEED),
                collision_distance=COLLISION_DISTANCE,
                all_obs=create_obstacles(_load_base_obstacles_config()))
            # UR5Group only sets this in setup(); before the first task is
            # loaded there are simply no obstacles.
            self.ur5_group.active_obs = {}
            # One marker per possible arm, colored to match that arm --
            # same pattern as environment/tasks.py: TaskManager.targets_visuals.
            # Parked far away (one slot apart) until a task assigns them a
            # real pose.
            self.target_markers = [
                Target(pose=[[i, 5, 0], [0, 0, 0, 1]], color=ur5.color)
                for i, ur5 in enumerate(self.ur5_group.all_controllers)
            ]
            # Each arm link's own color, so highlighting can fade an arm
            # and restore it exactly. {body_id: [(link, rgb), ...]}
            self._arm_link_colors = {
                ur5.body_id: [(shape[1], list(shape[7][:3]))
                              for shape in p.getVisualShapeData(ur5.body_id)]
                for ur5 in self.ur5_group.all_controllers
            }

        self.camera = dict(render.DEFAULT_CAMERA)

        # idle -> nothing loaded/planned yet
        # running -> a background planning thread is active
        # playback -> a path was found (frames holds it)
        # failed -> planning finished with no path
        self.mode = 'idle'
        # Every status/algorithm message (and anything passed to log()) is
        # also appended here, for the message log under the Run button.
        self._log_entries = []  # [(HH:MM:SS, text), ...], oldest first
        # Messages ever logged; unlike len(_log_entries) it keeps growing
        # once the list is capped, so pollers can tell something is new.
        self._log_total = 0
        self._log_lock = threading.Lock()
        self._status_message = 'No task loaded.'
        self.current_task = None
        self.frames = []
        self.obstacle_frames = None  # per-frame {obs_id: pos}, or None
        self._goal_tolerances = goal_tolerances()  # (position, orientation)
        self.frame_index = 0
        self.playing = False
        self.playback_speed = PLAYBACK_SPEEDS[0]
        self.show_path = False  # "Show trajectories"
        self.show_labels = False
        self.show_collision = True  # "Highlight collided links"
        self.show_caption = False
        self.task_name = None  # '<folder>/<filename>', for the caption
        # Keys (arm_key/obstacle_key) of highlighted objects; when any are
        # set, everything else is faded to HIGHLIGHT_DIM.
        self.highlight = set()
        # The last run's collision, if it had one: object keys, the
        # (arm index, link index) pairs in contact, and the frame it was
        # detected at. From that frame on, those links/obstacles are red.
        self.collided = []
        self.collided_links = set()
        self.collision_frame = None
        self._showing_collision = False
        self._showing_reached = frozenset()  # arm indices shown green
        self._path_marker_bodies = {}  # frame_idx -> [body_id, ...]
        self._path_visual_shapes = None
        self._obstacle_trails = ObstacleTrails()
        self._run_thread = None
        # The algorithm picked in the dropdown. Its adapter is created and
        # prepared (worker started, checkpoint loaded) in the background as
        # soon as it is picked, then reused for every run.
        self.algorithm_key = None
        self._algorithm_status = 'No algorithm selected.'
        self._adapter = None
        self._running_adapter = None
        self._preparing = False  # the selected adapter is still loading
        self._adapter_lock = threading.Lock()
        self._playback_thread = None

    def log(self, text):
        with self._log_lock:
            self._log_entries.append((time.strftime('%H:%M:%S'), text))
            del self._log_entries[:-MAX_LOG_ENTRIES]
            self._log_total += 1

    def log_entries(self):
        """(total messages ever logged, the latest MAX_LOG_ENTRIES)."""
        with self._log_lock:
            return self._log_total, list(self._log_entries)

    @property
    def status_message(self):
        return self._status_message

    @status_message.setter
    def status_message(self, text):
        self._status_message = text
        self.log(text)

    @property
    def algorithm_status(self):
        return self._algorithm_status

    @algorithm_status.setter
    def algorithm_status(self, text):
        self._algorithm_status = text
        self.log(text)

    def set_camera(self, **kwargs):
        with self.lock:
            self.camera.update(kwargs)

    def render_jpeg(self, width=render.DEFAULT_WIDTH, height=render.DEFAULT_HEIGHT):
        """The current view, re-rendered only if the scene, camera or
        labels changed since the last call (else the cached image)."""
        with self.lock._lock:  # reading only: don't count as a change
            caption = tuple(self._caption()) if self.show_caption else ()
            # The caption's camera note is covered by the camera in the key.
            key = (self.lock.version, tuple(sorted(self.camera.items())),
                   self.show_labels, caption, width, height)
            if key == self._last_render[0]:
                return self._last_render[1]
            labels = self._labels() if self.show_labels else ()
            jpeg = render.render_jpeg(
                self.camera, width=width, height=height, labels=labels,
                occluders=self._label_occluders(),
                caption=caption, caption_note=self._caption_note())
            self._last_render = (key, jpeg)
            return jpeg

    def render_image(self):
        """The current view as a PIL image, as the stream shows it."""
        with self.lock._lock:
            return render.render_image(
                self.camera,
                labels=self._labels() if self.show_labels else (),
                occluders=self._label_occluders(),
                caption=self._caption() if self.show_caption else (),
                caption_note=self._caption_note())

    def render_all_frames(self):
        """Every frame of the trajectory as PIL images, rendered with the
        current settings, or None if there is none. Pauses playback and
        returns to the current frame afterwards; the live view holds still
        meanwhile."""
        with self.lock:
            if not self.frames:
                return None
            self.playing = False
            current = self.frame_index
            self.log(f'Exporting {len(self.frames)} frames...')
            try:
                images = []
                for index in range(len(self.frames)):
                    self._apply_frame(index)
                    images.append(self.render_image())
                return images
            finally:
                self._apply_frame(current)

    def frame_seconds(self):
        """Time per frame at the current playback speed."""
        return 1.0 / (PLAYBACK_FPS * self.playback_speed)

    def object_keys(self):
        """([arm keys], [obstacle keys]) of the loaded task."""
        return ([arm_key(i) for i in range(
                    len(self.ur5_group.active_controllers))],
                [obstacle_key(obs_id) for obs_id in self.ur5_group.active_obs])

    def _labels(self):
        """Name tags for render.draw_labels: arms at their base, obstacles
        just above their top; only the highlighted ones, if any are."""
        labels = []
        for i, ur5 in enumerate(self.ur5_group.active_controllers):
            key = arm_key(i)
            if self._alpha_factor(key) < 1:
                continue
            x, y, z = ur5.get_pose()[0]
            labels.append({'text': object_label(key),
                           'position': [x, y, z],
                           'color': ur5.color,
                           'body': ur5.body_id,
                           'probe': [x, y, z + 0.05]})  # inside the base
        for obs_id, obs in self.ur5_group.active_obs.items():
            key = obstacle_key(obs_id)
            if self._alpha_factor(key) < 1:
                continue
            x, y, z = obs.get_position()
            labels.append({'text': object_label(key),
                           'position': [x, y, z + obs.radius],
                           'body': obs.body_id,
                           'probe': [x, y, z]})
        return labels

    def _label_occluders(self):
        """Body ids that hide a label's object when in front of it: arms
        and obstacles not faded by highlighting. Faded ones, goal markers
        and trails are see-through."""
        bodies = {ur5.body_id for i, ur5 in enumerate(
                      self.ur5_group.active_controllers)
                  if self._alpha_factor(arm_key(i)) == 1}
        bodies.update(obs.body_id for obs_id, obs
                      in self.ur5_group.active_obs.items()
                      if self._alpha_factor(obstacle_key(obs_id)) == 1)
        return bodies

    def set_show_labels(self, enabled):
        self.show_labels = enabled

    def set_show_caption(self, enabled):
        self.show_caption = enabled

    def _caption_note(self):
        """The caption's small grey last line: the camera view, as the
        sliders show it, plus its center once moved off the default."""
        if not self.show_caption:
            return None
        camera = self.camera
        note = (f"yaw: {camera['yaw']:.0f}, pitch: {camera['pitch']:.0f}, "
                f"distance: {camera['distance']:.1f}")
        target = [float(v) for v in camera['target']]
        if target != render.DEFAULT_CAMERA['target']:
            note += ', center: ({:.1f}, {:.1f}, {:.1f})'.format(*target)
        return note

    def _caption(self):
        """Lines of the "Show caption" box: planner, task, frame."""
        planner = (_algorithm_label(self.algorithm_key)
                   if self.algorithm_key else '(none)')
        frame = (f'{self.frame_index + 1}/{len(self.frames)}'
                 if self.frames else '-')
        return [f'Planner: {planner}',
                f'Task: {self.task_name or "(none)"}',
                f'Frame: {frame}']

    def set_show_collision(self, enabled):
        with self.lock:
            if enabled == self.show_collision:
                return
            self.show_collision = enabled
            self._apply_object_colors()

    def set_highlight(self, keys):
        with self.lock:
            keys = set(keys)
            if keys == self.highlight:
                return
            self.highlight = keys
            self._apply_highlight()

    def _alpha_factor(self, key):
        if not self.highlight or key in self.highlight:
            return 1.0
        return HIGHLIGHT_DIM

    def _clear_collision(self):
        self.collided = []
        self.collided_links = set()
        self.collision_frame = None

    def _apply_highlight(self):
        """Fades everything not highlighted -- arms, their goal markers and
        obstacles -- and hides their trails, or restores it all when
        nothing is highlighted."""
        self._apply_object_colors()
        # Rebuild the trails for the objects now shown.
        self._clear_path_markers()
        self._path_visual_shapes = None
        self._sync_path_markers()

    def _apply_object_colors(self):
        """Sets every arm link, obstacle and goal marker to its own color
        -- red if in contact, the current frame is at or past the
        collision and show_collision is on -- faded by _alpha_factor.

        Arms and obstacles are drawn opaque unless faded, although the
        end effector (alpha 0.5) and obstacles (0.9) are translucent
        elsewhere: PyBullet's OpenGL renderer draws translucent objects
        after opaque ones in a fixed order, not back to front from the
        camera, and without writing depth -- so a goal marker or red end
        effector behind an obstacle could be drawn over it. Opaque objects
        are depth-tested, so they always cover what is behind them.

        An arm whose end effector is within tolerance of its goal (see
        _arms_reached) has its goal marker and end effector green: a path
        that ends exactly on the goal (OMPL) leaves the marker inside the
        opaque end effector, which hides it."""
        in_collision = self._in_collision()
        reached = self._arms_reached()
        self._showing_collision = in_collision
        self._showing_reached = reached
        for i, marker in enumerate(
                self.target_markers[:len(self.ur5_group.active_controllers)]):
            marker.touched() if i in reached else marker.normal()
        for i, ur5 in enumerate(self.ur5_group.active_controllers):
            factor = self._alpha_factor(arm_key(i))
            for link, rgb in self._arm_link_colors[ur5.body_id]:
                if in_collision and (i, link) in self.collided_links:
                    rgb = COLLISION_RGB
                elif i in reached and link == ur5.EEF_LINK_INDEX:
                    rgb = REACHED_RGB
                p.changeVisualShape(ur5.body_id, link,
                                    rgbaColor=list(rgb) + [factor])
        for obs_id, obs in self.ur5_group.active_obs.items():
            key = obstacle_key(obs_id)
            rgb = (COLLISION_RGB if in_collision and key in self.collided
                   else obs.color[:3])
            p.changeVisualShape(obs.body_id, -1, rgbaColor=list(rgb) + [
                self._alpha_factor(key)])
        self._apply_target_highlight()

    def _in_collision(self):
        """Whether collided links/obstacles should be shown red now."""
        return (self.show_collision and self.collision_frame is not None
                and bool(self.frames)
                and self.frame_index >= self.collision_frame)

    def _arms_reached(self):
        """Indices of the arms whose end effector is at its goal in the
        current pose -- within the position and orientation tolerances
        the RL environment counts a goal as reached with, checked the
        same way (BaseEnv.check_ur5_reached_target)."""
        if self.current_task is None:
            return frozenset()
        pos_tol, orn_tol = self._goal_tolerances
        reached = set()
        for i, (ur5, target) in enumerate(zip(
                self.ur5_group.active_controllers,
                self.current_task.target_eff_poses)):
            pos, orn = ur5.get_end_effector_pose()
            pos_residual = math.dist(pos, target[0])
            # Smallest rotation between the two orientations.
            dot = abs(sum(a * b for a, b in zip(orn, target[1])))
            orn_residual = 2 * math.acos(min(1.0, dot))
            if pos_residual < pos_tol and orn_residual < orn_tol:
                reached.add(i)
        return frozenset(reached)

    def _apply_target_highlight(self):
        # Target.normal()/touched() only recolor on a state change, so the
        # faded alpha is reapplied on top of whichever color is current.
        for i, marker in enumerate(
                self.target_markers[:len(self.ur5_group.active_controllers)]):
            if marker._mode == Target.TOUCHED:
                rgba = [0.4, 1.0, 0.4, 0.8]  # as in Target.touched()
            else:
                rgba = list(marker.color[:3]) + [0.5]  # as in Target.normal()
            rgba[3] *= self._alpha_factor(arm_key(i))
            p.changeVisualShape(marker.body_id, -1, rgbaColor=rgba)

    def load_task(self, task, announce=True, name=None):
        """announce=False skips the "Loaded task" log line, for reloads
        right before a run (whose own message names the task). name:
        '<folder>/<filename>' for the caption; None keeps the current one."""
        with self.lock:
            self.current_task = task
            if name is not None:
                self.task_name = name
            self.frames = []
            self.obstacle_frames = None
            self._goal_tolerances = goal_tolerances()
            self._clear_collision()
            self.frame_index = 0
            self.mode = 'idle'
            message = (f'Loaded task "{task.id}" ({task.ur5_count} arm(s), '
                       f'{len(task.obstacles)} obstacle(s)).')
            if announce:
                self.status_message = message
            else:
                self._status_message = message
            self.ur5_group.setup(
                task.base_poses, task.start_config, obs_conf=task.obstacles)
            self._update_target_markers(task.target_eff_poses)
            # A new task starts with nothing highlighted; this also
            # restores anything faded in the previous task.
            self.highlight = set()
            self._apply_highlight()

    def _update_target_markers(self, target_eff_poses):
        for i, marker in enumerate(self.target_markers):
            if i < len(target_eff_poses):
                marker.set_pose(target_eff_poses[i])
            else:
                marker.set_pose([[i, 5, 0], [0, 0, 0, 1]])
            marker.normal()  # recolored by _apply_object_colors

    def set_show_path(self, enabled):
        with self.lock:
            self.show_path = enabled
            if enabled:
                self._path_visual_shapes = None
            self._sync_path_markers()

    def _apply_frame(self, frame_index):
        """Central point for moving to a given frame: applies joint values
        (and obstacle positions, if they move), updates each arm's reached
        (green) vs normal color, and (if enabled) updates the path
        trail up to this frame. seek(), _run_worker(), and
        _playback_worker() all go through this so those three stay in sync
        everywhere the current frame can change."""
        self.frame_index = frame_index
        self.ur5_group.set_joint_positions(self.frames[frame_index])
        if self.obstacle_frames:
            for obs_id, pos in self.obstacle_frames[frame_index].items():
                self.ur5_group.active_obs[obs_id].set_position(pos)
        if (self._in_collision() != self._showing_collision
                or self._arms_reached() != self._showing_reached):
            # entering/leaving the collision, or an arm its goal
            self._apply_object_colors()
        elif self.highlight:
            self._apply_target_highlight()
        self._sync_path_markers()

    def _clear_path_markers(self):
        for body_ids in self._path_marker_bodies.values():
            path_markers.remove_markers(body_ids)
        self._path_marker_bodies = {}
        self._obstacle_trails.clear()

    @property
    def path_marker_ids(self):
        return [bid for ids in self._path_marker_bodies.values() for bid in ids]

    def _sync_path_markers(self):
        """Diffs the desired marker set (see path_markers.desired_frame_indices)
        against what's already in the scene. A single-frame forward step
        during normal playback costs one add and at most one remove --
        not a full rebuild -- which is what keeps this cheap enough to run
        on every playback tick (~0.1s per *full* rebuild was measured at
        the marker cap, comparable to the whole tick interval)."""
        if not self.show_path or not self.frames:
            self._clear_path_markers()
            return
        self._obstacle_trails.sync(
            self.obstacle_frames, self.frame_index, self.ur5_group.active_obs,
            lambda obs_id: self._alpha_factor(obstacle_key(obs_id)) == 1)
        desired = path_markers.desired_frame_indices(self.frame_index)
        existing = set(self._path_marker_bodies.keys())
        for idx in existing - desired:
            path_markers.remove_markers(self._path_marker_bodies.pop(idx))
        to_add = desired - existing
        if to_add:
            if self._path_visual_shapes is None:
                self._path_visual_shapes = path_markers.make_visual_shapes(
                    self.ur5_group,
                    [self._alpha_factor(arm_key(i)) == 1 for i in range(
                        len(self.ur5_group.active_controllers))])
            for idx in to_add:
                self._path_marker_bodies[idx] = path_markers.create_markers_for_frame(
                    self.frames, idx, self.ur5_group, self._path_visual_shapes)

    @property
    def busy(self):
        """True while the selected algorithm is loading or a task is being
        planned -- the selection controls are disabled meanwhile."""
        return self._preparing or (
            self._run_thread is not None and self._run_thread.is_alive())

    def select_algorithm(self, algorithm_key):
        """Makes `algorithm_key` the current algorithm and prepares its
        adapter in the background, closing the previously selected one
        (once any run still using it has finished). Returns whether the
        selection changed."""
        with self._adapter_lock:
            if algorithm_key == self.algorithm_key:
                return False
            old = self._adapter
            self.algorithm_key = algorithm_key
            self._adapter = (create_adapter(algorithm_key)
                             if algorithm_key else None)
            adapter = self._adapter
            self.algorithm_status = (
                f'Loading {_algorithm_label(algorithm_key)}...'
                if adapter is not None else 'No algorithm selected.')
            self._preparing = adapter is not None
        if old is not None and old is not self._running_adapter:
            old.close()
        if adapter is None:
            return True
        threading.Thread(
            target=self._prepare_worker, args=(algorithm_key, adapter),
            daemon=True).start()
        return True

    def _prepare_worker(self, algorithm_key, adapter):
        label = _algorithm_label(algorithm_key)
        start_time = time.monotonic()
        try:
            adapter.prepare()
        except Exception as exc:
            if self._adapter is adapter:
                self.algorithm_status = f'Failed to load {label}: {exc}'
                self._preparing = False
            return
        if self._adapter is adapter:
            elapsed = time.monotonic() - start_time
            self.algorithm_status = f'{label} ready (loaded in {elapsed:.1f}s).'
            self._preparing = False

    def run(self, algorithm_key):
        if self.current_task is None:
            self.status_message = 'Load a task first.'
            return
        if self._run_thread is not None and self._run_thread.is_alive():
            return
        self._run_thread = threading.Thread(
            target=self._run_worker, args=(algorithm_key,), daemon=True)
        self._run_thread.start()

    def _run_worker(self, algorithm_key):
        task = self.current_task
        self.select_algorithm(algorithm_key)  # no-op if already selected
        with self._adapter_lock:
            adapter = self._running_adapter = self._adapter
        with self.lock:
            self.mode = 'running'
            self._clear_collision()
            self.status_message = (
                f'Planning task "{task.id}" with '
                f'{_algorithm_label(algorithm_key)}...')
        try:
            self._plan_with(adapter, task)
        finally:
            with self._adapter_lock:
                self._running_adapter = None
                replaced = adapter is not self._adapter
            if replaced:  # another algorithm was picked while this ran
                adapter.close()

    def _plan_with(self, adapter, task):
        start_time = time.monotonic()
        try:
            if adapter.uses_sim:
                with self.lock:
                    path = list(adapter.plan(task, self.ur5_group))
            else:
                # Plans in its own process -- keep the lock free so the
                # video feed keeps streaming while it runs.
                path = list(adapter.plan(task, self.ur5_group))
        except Exception as exc:
            elapsed = time.monotonic() - start_time
            with self.lock:
                self.mode = 'failed'
                self.status_message = f'Planner error after {elapsed:.1f}s: {exc}'
            return
        elapsed = time.monotonic() - start_time
        with self.lock:
            if self.current_task is not task:
                return  # a different task was loaded while planning
            if not path:
                self.mode = 'failed'
                self.status_message = (adapter.result_message
                                       or f'No path found ({elapsed:.1f}s).')
                self.ur5_group.setup(
                    task.base_poses, task.start_config,
                    obs_conf=task.obstacles)
                return
            adapter.check_replay(task, path, self.ur5_group)
            self.frames = path
            self.obstacle_frames = adapter.obstacle_frames
            self.collided = list(adapter.collided)
            self.collided_links = set(adapter.collided_links)
            self.collision_frame = adapter.collision_frame
            self._clear_path_markers()
            self._path_visual_shapes = None
            self._apply_frame(0)
            self.mode = 'playback'
            self.status_message = (
                adapter.result_message
                or f'Path found: {len(path)} waypoints in {elapsed:.1f}s.')
        self.play()

    def seek(self, frame_index):
        with self.lock:
            if not self.frames:
                return
            frame_index = max(0, min(frame_index, len(self.frames) - 1))
            self._apply_frame(frame_index)

    def step(self, delta):
        """Moves `delta` frames, leaving playback playing or paused as it
        was."""
        self.seek(self.frame_index + delta)

    def replay(self):
        """Restarts from the first frame and plays, whatever the state."""
        self.seek(0)
        self.play()

    def toggle_play(self):
        if self.playing:
            self.pause()
        else:
            self.play()

    def cycle_speed(self):
        index = PLAYBACK_SPEEDS.index(self.playback_speed)
        self.playback_speed = PLAYBACK_SPEEDS[
            (index + 1) % len(PLAYBACK_SPEEDS)]

    def play(self):
        if self.mode != 'playback' or not self.frames:
            return
        if self.frame_index >= len(self.frames) - 1:
            # Restart from the beginning if already at the end.
            self.seek(0)
        if self.playing:
            return
        self.playing = True
        if self._playback_thread is None or not self._playback_thread.is_alive():
            self._playback_thread = threading.Thread(
                target=self._playback_worker, daemon=True)
            self._playback_thread.start()

    def pause(self):
        self.playing = False

    def _playback_worker(self):
        next_tick = time.monotonic()
        while self.playing:
            with self.lock:
                if not self.playing:
                    break
                if self.frame_index >= len(self.frames) - 1:
                    self.playing = False
                    break
                self._apply_frame(self.frame_index + 1)
            # Pace by absolute schedule, not a fixed per-iteration sleep --
            # otherwise time spent waiting for the lock (e.g. behind a
            # render_jpeg call) compounds into ever-growing delay instead of
            # being caught up on the next tick.
            # Read every tick, so a speed change applies immediately.
            next_tick += self.frame_seconds()
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)


def _algorithm_label(algorithm_key):
    return dict(list_adapters()).get(algorithm_key, algorithm_key)


# Single global instance -- see module docstring.
sim_manager = SimulationManager()
