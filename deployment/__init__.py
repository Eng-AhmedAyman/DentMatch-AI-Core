"""
==============================================================================
PACKAGE: deployment
DESCRIPTION:
Production Deployment Modules for the Healthy Smile AI Engine.
This package orchestrates the Master Pipeline (3-Stage AI Integration) 
and the Explainable AI (Grad-CAM) visualization engine for the clinical dashboard.
==============================================================================
"""

__version__ = "1.0.0"
__author__ = "Ahmed Ayman"
__status__ = "Production Ready"

# Expose core classes for clean imports in app.py and api.py
from .master_pipeline import DentalAI_System