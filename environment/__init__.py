import importlib

# These three are ray/torch-free (UR5, TaskLoader/Task, and the utils helpers
# no longer import torch at module scope), so they stay eager just like
# before. Keeping `from .UR5 import UR5` eager also matters for correctness:
# `.tasks` internally does `from .UR5 import Robotiq2F85Target`, which makes
# Python auto-bind `environment.UR5` to the *submodule* as a side effect of
# that import; importing the UR5 class here first (and letting it get
# re-imported, a no-op, later) ensures `environment.UR5` ends up being the
# class, not the submodule.
from .UR5 import UR5
from .tasks import TaskLoader, TasksExhausted
import environment.utils as utils

__all__ = [
    'BaseEnv',
    'BenchmarkEnv',
    'RRTSupervisionEnv',
    'RRTWrapper',
    'TaskLoader',
    'TasksExhausted',
    'utils',
    'RealTimeEnv',
    'UR5'
]

# Attribute -> (submodule, attribute-on-submodule). Resolved lazily on first
# access (PEP 562) so that importing anything from this package (e.g.
# `environment.rrt.ur5_group`, which only needs `environment.UR5`) doesn't
# force ray/torch to be importable just because some *other* attribute here
# needs them. Each of these still imports ray and/or torch transitively
# (baseEnv, benchmarkEnv, rrtSupervisionEnv, realTimeEnv, and RRTWrapper via
# its @ray.remote decorator), so they're only actually imported when a
# caller asks for them by name.
_LAZY_ATTRS = {
    'BenchmarkEnv': ('.benchmarkEnv', 'ParallelBenchmarkEnv'),
    'BaseEnv': ('.baseEnv', 'ParallelBaseEnv'),
    'RRTSupervisionEnv': ('.rrtSupervisionEnv', 'RRTSupervisionEnv'),
    'RRTWrapper': ('.rrt', 'RRTWrapper'),
    'RealTimeEnv': ('.realTimeEnv', 'RealTimeEnv'),
}


def __getattr__(name):
    if name in _LAZY_ATTRS:
        module_name, attr = _LAZY_ATTRS[name]
        module = importlib.import_module(module_name, __name__)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
