"""Register all SQLAlchemy tables in every entry point using this package."""

from .account import Account
from .dataset import Dataset
from .review import Review
from .run import Run

__all__ = ['Account', 'Dataset', 'Review', 'Run']
