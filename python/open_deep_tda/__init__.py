"""Open Deep-TDA: an independent, experimental topology-regularized reducer."""
from .config import TDAConfig
from .estimator import DeepTDA

__all__ = ["DeepTDA", "TDAConfig"]
__version__ = "0.3.0"
