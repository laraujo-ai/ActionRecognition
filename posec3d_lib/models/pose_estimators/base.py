from typing import Protocol, Any, List


class BasePoseEstimator(Protocol):
    """Protocol interface for pose estimation models.
    
    Defines the contract that all pose estimators must implement for detecting
    human poses in video frames. Pose estimators are the first stage in the
    action recognition pipeline, providing keypoint coordinates and confidence
    scores for downstream processing.
    
    The typical pose estimation workflow:
    1. Preprocess input frames (resize, normalize)
    2. Run pose detection model inference
    3. Post-process outputs (NMS, coordinate conversion, filtering)
    """
    def preprocess(self, data: Any) -> Any:
        """Preprocess input data for pose estimation.
        
        Args:
            data: Input data (typically video frames or images)
            
        Returns:
            Any: Preprocessed data ready for model inference
        """
        ...
    def predict(self, data: Any) -> Any:
        """Run pose estimation inference on preprocessed data.
        
        Args:
            data: Preprocessed input data from preprocess() method
            
        Returns:
            Any: Raw model outputs (typically pose predictions)
        """
        ...
    def postprocess(self, output: Any, **kwargs) -> List:
        """Post-process model outputs to extract pose information.
        
        Args:
            output: Raw model outputs from predict() method
            **kwargs: Additional parameters for post-processing
            
        Returns:
            List: List of pose detection results with keypoints and scores
        """
        ...
