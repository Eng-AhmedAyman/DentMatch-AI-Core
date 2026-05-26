"""
==============================================================================
PACKAGE: deployment
DESCRIPTION:
    Production Deployment Modules for the Healthy Smile AI Engine.
    This package orchestrates the Master Pipeline (3-Stage AI Integration)
    and the Explainable AI (Grad-CAM) visualization engine for the clinical
    dashboard.

    NOTE: No models are imported at package level to avoid loading all three
    AI models as a side-effect of any indirect import. Each module imports
    directly from master_pipeline or explainability as needed.
==============================================================================
"""

__version__ = "1.0.0"
__author__ = "Ahmed Ayman"
__status__ = "Production Ready"
