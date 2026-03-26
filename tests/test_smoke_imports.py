import importlib

MODULES = [
    "config.defaults",
    "data.priors",
    "data.splits",
    "sim.stimulus",
    "models.param_transforms",
]

def test_smoke_imports():
    for name in MODULES:
        importlib.import_module(name)
