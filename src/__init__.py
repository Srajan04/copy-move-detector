"""
Visual Forensics and Copy-Move Forgery Detection Package

This package provides a comprehensive solution for detecting copy-move forgeries
in digital images using SIFT features and geometric verification.
"""

__version__ = "1.0.0"
__author__ = "CVAI Project"

from .forgery_detector import CopyMoveForgeryDetector
from .batch_processor import BatchProcessor

__all__ = ["CopyMoveForgeryDetector", "BatchProcessor"]
