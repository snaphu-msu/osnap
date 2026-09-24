from importlib import import_module

__all__ = ['config', 'info', 'odb', 'progenitor', 'trajectories', 'units', 'viz']


def __getattr__(name):
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(name)
