from typing import Protocol, Sequence, Any
from posec3d_lib.utils.composer import IComposer


class BasePreprocessor(Protocol):
    """Protocol interface for data preprocessors in the PoseC3D pipeline.
    
    Defines the contract that all preprocessors must implement for transforming
    raw pose detection data into the format expected by recognition models.
    
    Preprocessors typically handle:
    1. Data format conversion and normalization
    2. Temporal sequence processing
    3. Spatial transformations and augmentations
    4. Model-specific input formatting
    """
    def __init__(self, transforms_pipeline):
        """Initialize the preprocessor with a transformation pipeline.
        
        Args:
            transforms_pipeline: Composer or pipeline containing the sequence
                of transformations to apply to input data
        """
        ...
    def process(self, data: Any) -> Any:
        """Process input data through the preprocessing pipeline.
        
        Args:
            data: Raw input data (typically pose detection results)
            
        Returns:
            Any: Preprocessed data ready for model inference
        """
        ...
