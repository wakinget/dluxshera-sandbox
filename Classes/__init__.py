name = "Classes"
__version__ = "0.1.0"

# Import as modules
from . import optical_systems
from . import oneoverf
from . import utils
from . import optimization
from . import modeling


# Add to __all__
modules = [
    optical_systems,
    oneoverf,
    utils,
    optimization,
    modeling
]

# __all__ = [module.__all__ for module in modules] # Threw an error to do with nested lists
__all__ = sum((module.__all__ for module in modules), []) # This should work?
