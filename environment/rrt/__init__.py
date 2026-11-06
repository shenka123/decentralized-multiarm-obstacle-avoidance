import importlib

__all__ = [
    'RRTWrapper'
]


def __getattr__(name):
    # RRTWrapper needs ray importable (it's a @ray.remote class), which is a
    # heavy dependency not every consumer of environment.rrt has installed
    # (e.g. environment.rrt.ur5_group / rrt_connect are ray-free). Deferring
    # the import here means those stay importable without ray.
    if name == 'RRTWrapper':
        module = importlib.import_module('.rrtWrapper', __name__)
        value = module.RRTWrapper
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
