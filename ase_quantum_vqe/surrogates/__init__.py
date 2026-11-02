"""Surrogate models for on-the-fly active learning (FALCON-compatible)."""
from .distance_gpr import DistanceGPR, DistanceDescriptor

__all__ = ["DistanceGPR", "DistanceDescriptor"]
