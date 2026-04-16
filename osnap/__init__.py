from importlib import import_module

__all__ = ['load_data',
           'config',
           'plotting',
           'stitching',
           'save_data',
           'tools',
           'nucleo',
           'heger02_composition',
           ]


def __getattr__(name):
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(name)
