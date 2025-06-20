"""
Performance Optimization with Wellbeing Focus

Performance optimization system that prioritizes user wellbeing, emotional safety,
and healing outcomes over traditional engagement or speed metrics.

Core Philosophy: "Fast enough to care, efficient enough to heal."
"""

from .wellbeing_optimizer import WellbeingOptimizer, OptimizationLevel, CareContext
from .performance_monitor import PerformanceMonitor, WellbeingMetric, ResponsePriority
from .resource_allocator import ResourceAllocator, CareQueue, AllocationStrategy
from .caring_profiler import CaringProfiler, ProfilerMode, WellbeingProfile

__all__ = [
    'WellbeingOptimizer',
    'OptimizationLevel',
    'CareContext',
    'PerformanceMonitor', 
    'WellbeingMetric',
    'ResponsePriority',
    'ResourceAllocator',
    'CareQueue',
    'AllocationStrategy',
    'CaringProfiler',
    'ProfilerMode',
    'WellbeingProfile'
]