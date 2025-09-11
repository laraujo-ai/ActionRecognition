import logging
import numpy as np
import time

from posec3d_lib.models.recognizers.base import IRecognizer
from posec3d_lib.models.functions import softmax
from posec3d_lib.models.preprocessors.posec3d_preprocessor import Posec3dPreprocessor
logger = logging.getLogger(__name__)


class Posec3DRecognizer(IRecognizer):
    """PoseC3D action recognition model for classifying human actions from pose sequences.
    
    This class implements action recognition using the PoseC3D architecture, which processes
    temporal sequences of human pose keypoints to classify actions. It uses ONNX Runtime
    for optimized inference and supports the NTU RGB+D 60 action dataset.
    
    The model processes pose sequences through:
    1. Preprocessing pose keypoints into 3D heatmap volumes
    2. Running inference through the PoseC3D ONNX model
    3. Post-processing outputs to get action class predictions
    
    Attributes:
        model_engine: ONNX Runtime inference session
        preprocessor: Posec3dPreprocessor for input data transformation
        input_name: Name of the model's input tensor
        label_map: List of action class names for prediction mapping
    """

    def __init__(
        self, preprocessor: Posec3dPreprocessor, model_engine, label_map_path: str
    ):
        """Initialize PoseC3D action recognizer.
        
        Args:
            preprocessor: Posec3dPreprocessor instance for data transformation
            model_engine: ONNX Runtime inference session for the PoseC3D model
            label_map_path: Path to text file containing action class names
            
        Raises:
            Exception: If model initialization or label map loading fails
        """
        try:
            self.model_engine = model_engine
            self.preprocessor = preprocessor
            self.input_name = self.model_engine.get_inputs()[0].name
            self.label_map = [x.strip() for x in open(label_map_path).readlines()]
        except Exception as e:
            logger.error(f"Failed to initialize Posec3DRecognizer: {e}")
            raise

    def inference(self, data):
        """Run action recognition inference on pose data.
        
        Args:
            data: Dictionary containing pose results and image metadata:
                - pose_results: List of pose detections per frame
                - img_shape: Tuple of (height, width) for the video frames
                
        Returns:
            str: Predicted action class name
            
        Raises:
            Exception: If preprocessing or model inference fails
        """
        try:
            input_ = self.preprocessor.process(data)
            start_inference = time.time()
            outputs = self.model_engine.run(None, {self.input_name: input_})
            end_inference = time.time()
            logger.info(f"time spent at inference for the action onnx model: {end_inference - start_inference:.2f} ms")
            return self.post_process_results(outputs)
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise

    def post_process_results(self, outputs: list):
        """Post-process model outputs to get action prediction.
        
        Applies softmax to model outputs, averages across temporal clips,
        and returns the predicted action class name.
        
        Args:
            outputs: List of model output tensors from ONNX inference
            
        Returns:
            str: Predicted action class name from the label map
        """
        batched_results = []
        for output in outputs:
            out = softmax(output, axis=1).mean(axis=0)
            out = out.argmax().item(0)
            predicted_action = self.label_map[out]
            batched_results.append(predicted_action)

        return batched_results[
            0
        ]  # will return only the first video result for now, after if needed we can use batch


class Posec3DFeatureExtractor(IRecognizer):
    """PoseC3D feature extraction model for extracting action features from pose sequences.
    
    This class uses a variant of the PoseC3D model to extract feature representations
    instead of performing classification. Useful for downstream tasks, similarity
    matching, or building custom classifiers on top of PoseC3D features.
    
    The feature extractor processes pose sequences through:
    1. Preprocessing pose keypoints into 3D heatmap volumes 
    2. Running inference through the feature extraction ONNX model
    3. Post-processing to return averaged feature vectors
    
    Attributes:
        model_engine: ONNX Runtime inference session for feature extraction
        preprocessor: Posec3dPreprocessor for input data transformation
        input_name: Name of the model's input tensor
    """

    def __init__(self, preprocessor, model_engine):
        """Initialize PoseC3D feature extractor.
        
        Args:
            preprocessor: Posec3dPreprocessor instance for data transformation
            model_engine: ONNX Runtime inference session for the feature extraction model
            
        Raises:
            Exception: If model initialization fails
        """
        try:
            self.model_engine = model_engine
            self.preprocessor = preprocessor
            self.input_name = self.model_engine.get_inputs()[0].name
        except Exception as e:
            logger.error(f"Failed to initialize Posec3DRecognizer: {e}")
            raise

    def inference(self, data):
        """Extract features from pose data.
        
        Args:
            data: Dictionary containing pose results and image metadata:
                - pose_results: List of pose detections per frame
                - img_shape: Tuple of (height, width) for the video frames
                
        Returns:
            np.ndarray: Feature vector representing the input pose sequence
            
        Raises:
            Exception: If preprocessing or model inference fails
        """
        try:
            input_ = self.preprocessor.process(data)
            outputs = self.model_engine.run(None, {self.input_name: input_})
            return self.post_process_results(outputs)

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise

    def post_process_results(self, outputs: list) -> np.ndarray:
        """Post-process model outputs to get feature vector.
        
        Averages the feature outputs across the temporal dimension to create
        a fixed-size feature representation of the input pose sequence.
        
        Args:
            outputs: List of model output tensors from ONNX inference
            
        Returns:
            np.ndarray: Averaged feature vector for the input pose sequence
        """
        features = outputs[0]  # Just the first batch for now
        feature_vector = features.mean(axis=0)
        feature_vector /= np.linalg.norm(feature_vector)
        return feature_vector
