""""Show trajectories" visualization: real geometry markers for the EEF trajectory.

PyBullet's addUserDebugLine is a GUI-only overlay and does not appear in
getCameraImage captures (see render.py), so the path is drawn as small,
translucent dash marks (short oriented cylinders) -- visual-only, no
collision shape -- placed at each arm's end-effector position along the
recorded frames, oriented along that arm's local direction of travel.

Only the trail from frame 0 up to the current frame is drawn, and it must
stay cheap on every single playback tick (~10/s): a naive
remove-everything-then-recreate-everything approach measured ~0.1s per
call at the marker cap, which is comparable to the whole playback interval
and reintroduces the sluggishness already fixed once (see sim_manager's
GUI/hardware-rendering change). So the marker set for a given frame index
is computed from a fixed frame stride (desired_frame_indices below), and
SimulationManager._sync_path_markers diffs the desired set against what's
already in the scene -- a single-frame forward step during normal
playback then costs one add and at most one remove, not a full rebuild.

The stride is a fixed frame count (FRAME_STRIDE), not a per-path count
cap: since playback advances one frame per fixed time tick, a constant
frame stride means each dash-to-dash gap covers the same wall-clock time,
so gap *distance* directly shows speed -- wide gaps where the arm moved
far in that time (fast), tight gaps where it barely moved (slow). A
length-normalized cap (resample to N markers regardless of path length)
would hide that by stretching/compressing spacing to fit a fixed count.
"""
import numpy as np
import pybullet as p

DASH_RADIUS = 0.008
DASH_LENGTH = 0.03
MARKER_ALPHA = 0.6
FRAME_STRIDE = 4


def desired_frame_indices(upto_index, stride=FRAME_STRIDE):
    """Frame 0, every `stride`-th frame, and upto_index itself.

    upto_index is always included so the trail's tip tracks the current
    frame exactly, even between stride boundaries.
    """
    if upto_index < 0:
        return set()
    indices = set(range(0, upto_index + 1, stride))
    indices.add(0)
    indices.add(upto_index)
    return indices


def make_visual_shapes(ur5_group, shown=None):
    """One cached dash-shaped visual shape per active arm, colored to match
    it. shown (one bool per arm, default all True): arms whose trail is
    drawn, e.g. only the highlighted ones; the others get None."""
    controllers = ur5_group.active_controllers
    if shown is None:
        shown = [True] * len(controllers)
    return [
        p.createVisualShape(
            p.GEOM_CYLINDER, radius=DASH_RADIUS, length=DASH_LENGTH,
            rgbaColor=(ctrl.color[0], ctrl.color[1], ctrl.color[2],
                       MARKER_ALPHA))
        if arm_shown else None
        for ctrl, arm_shown in zip(controllers, shown)
    ]


def _quat_aligning_z_to(direction):
    """Quaternion rotating the local +Z axis (a cylinder's long axis in
    PyBullet) to point along `direction`. Identity if direction is ~zero."""
    d = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(d)
    if norm < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    d /= norm
    z = np.array([0.0, 0.0, 1.0])
    dot = float(np.dot(z, d))
    if dot > 0.999999:
        return (0.0, 0.0, 0.0, 1.0)
    if dot < -0.999999:
        return (1.0, 0.0, 0.0, 0.0)  # 180 degree flip about X
    axis = np.cross(z, d)
    s = np.sqrt((1.0 + dot) * 2.0)
    inv_s = 1.0 / s
    return (axis[0] * inv_s, axis[1] * inv_s, axis[2] * inv_s, s * 0.5)


def create_markers_for_frame(frames, idx, ur5_group, visual_shapes):
    """Returns the body ids created for one frame index (one dash per
    active arm with a visual shape), oriented along that arm's direction
    of travel between the neighboring frames (falls back to a one-sided
    difference at either end of the path)."""
    prev_idx = idx - 1 if idx > 0 else idx
    next_idx = idx + 1 if idx < len(frames) - 1 else idx
    current_poses = ur5_group.forward_kinematics(frames[idx])
    prev_poses = (current_poses if prev_idx == idx
                  else ur5_group.forward_kinematics(frames[prev_idx]))
    next_poses = (current_poses if next_idx == idx
                  else ur5_group.forward_kinematics(frames[next_idx]))
    body_ids = []
    for arm_idx, pose in enumerate(current_poses):
        if visual_shapes[arm_idx] is None:
            continue
        direction = (np.array(next_poses[arm_idx][0])
                     - np.array(prev_poses[arm_idx][0]))
        body_ids.append(p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_shapes[arm_idx],
            basePosition=pose[0],
            baseOrientation=_quat_aligning_z_to(direction)))
    return body_ids


def remove_markers(body_ids):
    for body_id in body_ids:
        p.removeBody(body_id)
