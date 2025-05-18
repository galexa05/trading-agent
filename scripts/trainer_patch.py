"""
This module provides a direct monkey patch for the Accelerator class to handle version incompatibilities.
It patches the Accelerator's __init__ method to ignore unsupported parameters.
"""

import logging
from functools import wraps
from accelerate import Accelerator

logger = logging.getLogger(__name__)

# Store the original __init__ method
original_init = Accelerator.__init__

@wraps(original_init)
def patched_init(self, *args, **kwargs):
    """
    Patched __init__ method that ignores unsupported parameters
    """
    # Filter out problematic parameters
    filtered_kwargs = kwargs.copy()
    for param in list(filtered_kwargs.keys()):
        if param in ['use_seedable_sampler']:
            logger.warning(f"Removing unsupported parameter '{param}' from Accelerator")
            del filtered_kwargs[param]
    
    # Call the original __init__ with filtered kwargs
    return original_init(self, *args, **filtered_kwargs)

# Apply the monkey patch
Accelerator.__init__ = patched_init
logger.info("Accelerator.__init__ has been patched to handle version incompatibilities")
