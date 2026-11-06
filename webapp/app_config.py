"""webapp/config.json: settings for the webapp only -- separate from the
training/benchmark configs in configs/, which it never changes.

Keys (all optional):
    planner_order  dropdown labels, in the order to list them; planners not
                   listed follow alphabetically (see algorithms/registry.py)
    rl_config      config the RL rollout runs with (environment, observation
                   and network settings), default configs/evaluate.json --
                   the same one bulk_evaluate.py benchmarks with
    early_stop     {"threshold": float, "steps": int}, overriding rl_config's
                   environment.early_stop: an RL episode stops early once the
                   size of the policy's joint change (the norm of all arms'
                   actions in a control step, 0.05 s) stays below threshold
                   for `steps` control steps in a row
    ompl_playback  {"max_joint_step": float, "max_joint_accel": float or
                   null, "enabled": bool}: how OMPL paths are replayed
                   (see algorithms/ompl.retime). The path is resampled into
                   0.05 s frames, like RL control steps, with no joint
                   moving more than max_joint_step rad per frame (default
                   0.06). max_joint_accel (default 0.01 rad per frame, per
                   frame) ramps the speed up and down; null keeps it
                   constant. "enabled": false replays OMPL's own 200
                   waypoints unchanged

Read fresh on every call: planner_order applies on a page refresh, the RL
settings when an RL planner is next selected (that starts its worker),
ompl_playback on the next OMPL run.

The end-effector hit box is not set here: it comes from rl_config's
environment.ee_hitbox (configs/evaluate.json uses the drawn cylinder).
"""
import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent / 'config.json'
DEFAULT_RL_CONFIG = 'configs/evaluate.json'


def load():
    """The config as a dict, or {} if the file is missing or unreadable."""
    if not CONFIG_PATH.exists():
        return {}
    try:
        with open(CONFIG_PATH) as f:
            config = json.load(f)
    except (OSError, ValueError) as exc:
        print(f'[app_config] Ignoring {CONFIG_PATH}: {exc}')
        return {}
    if not isinstance(config, dict):
        print(f'[app_config] Ignoring {CONFIG_PATH}: not a JSON object')
        return {}
    return config

