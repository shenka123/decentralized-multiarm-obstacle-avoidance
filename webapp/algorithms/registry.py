"""Registry of available algorithm adapters, keyed by dropdown value.

Two families, labelled by where they come from:
    RL - <checkpoint>    one entry per checkpoint file in default/ and
                         runs/webapp/ (a runs/webapp/ file overrides a
                         default/ one with the same name)
    OMPL - <planner>     one entry per name in OMPL_PLANNERS

Dropdown order comes from "planner_order" in webapp/config.json (a list of
labels): listed planners first, in that order, then any others sorted
alphabetically by label. Labels listed there that don't exist are skipped.

default/ and runs/webapp/ are rescanned on every call, and the layout is rebuilt on every
page load (see app.py), so a checkpoint copied in there shows up after a
browser refresh -- no restart needed.

The original pybullet RRT-Connect adapter (rrt_connect.py) is no longer
listed; OMPL - RRTConnect is the same algorithm as the benchmark baseline.
"""
from .. import app_config
from .ompl import OMPLAdapter
from .rl import RLAdapter
from .wsl_worker import REPO_ROOT

OMPL_PLANNERS = ['BITstar', 'RRTConnect']
# default/ also holds the Baseline's configs; *.json files are skipped.
RL_CHECKPOINT_ROOTS = [REPO_ROOT / 'default', REPO_ROOT / 'runs' / 'webapp']


def _configured_order():
    """Labels from config.json's planner_order, or [] if unset."""
    order = app_config.load().get('planner_order', [])
    if not isinstance(order, list):
        return []
    return [label for label in order if isinstance(label, str)]


def _adapter_factories():
    """{key: (label, zero-arg factory)}"""
    factories = {}
    for root in RL_CHECKPOINT_ROOTS:
        if not root.exists():
            continue
        for ckpt in sorted(root.iterdir()):
            if (ckpt.is_file() and not ckpt.name.startswith('.')
                    and ckpt.suffix != '.json'):
                factories[f'rl:{ckpt.name}'] = (
                    f'RL - {ckpt.name}', lambda ckpt=ckpt: RLAdapter(ckpt))
    for name in OMPL_PLANNERS:
        factories[f'ompl:{name}'] = (
            f'OMPL - {name}', lambda name=name: OMPLAdapter(name))
    return factories


def list_adapters():
    """Returns [(key, label), ...] for populating a dropdown, in the
    configured order (see module docstring)."""
    adapters = [(key, label)
                for key, (label, _) in _adapter_factories().items()]
    rank = {label: i for i, label in enumerate(_configured_order())}
    listed = sorted((a for a in adapters if a[1] in rank),
                    key=lambda a: rank[a[1]])
    unlisted = sorted((a for a in adapters if a[1] not in rank),
                      key=lambda a: a[1].lower())
    return listed + unlisted


def create_adapter(key):
    return _adapter_factories()[key][1]()
