from typing import Protocol, Any


class IRecognizer(Protocol):
    """Protocol interface for action recognition models.
    
    Defines the contract that all action recognition models must implement.
    This includes both classification models (that return action class names)
    and feature extraction models (that return feature vectors).
    
    All recognizers should be able to:
    1. Initialize with a preprocessor and model engine
    2. Run inference on pose data to get predictions or features
    """
    def __init__(self, preprocessor, model_engine):
        """Initialize the recognizer with preprocessor and model engine.
        
        Args:
            preprocessor: Data preprocessor for transforming input data
            model_engine: ONNX Runtime inference session or similar model engine
        """
        ...
    def inference(self, data: Any) -> Any:
        """Run inference on input data.
        
        Args:
            data: Input data (typically pose sequences with metadata)
            
        Returns:
            Any: Model predictions (action class names or feature vectors)
        """
        ...
