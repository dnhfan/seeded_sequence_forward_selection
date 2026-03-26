"""
SFS - seeded forward selection.
     a hybird feature selection method using wrapper (forward selection) and filter (ensemble method to voting)
"""

from .forward_selection import SeededForwardSelection

__all__ = ["SeededForwardSelection"]
