from importlib.metadata import PackageNotFoundError, version

from .core import MorseCode

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "0.0.0"
__all__ = ["MorseCode", "__version__"]
