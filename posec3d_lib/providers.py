from types import ModuleType
import logging
import onnxruntime as ort

from posec3d_lib.utils.composer import Posec3dComposer, IComposer
from posec3d_lib.models.recognizers.Posec3D import (
    Posec3DRecognizer,
    IRecognizer,
    Posec3DFeatureExtractor,
)
from posec3d_lib.utils.parser import Posec3dConfigParser
from posec3d_lib.models.preprocessors.posec3d_preprocessor import (
    BasePreprocessor,
    Posec3dPreprocessor,
)
from posec3d_lib.models.pose_estimators.yolo_estimator import YoloPoseEstimator

logger = logging.getLogger(__name__)

# ONNX Runtime execution providers configuration for NVIDIA Jetson
# Prioritizes CUDA acceleration with optimized memory and performance settings
providers = [
    # TensorRT provider (commented out - can be enabled for additional optimization)
    # ('TensorrtExecutionProvider', {
    #     'device_id': 0,                       # Select GPU to execute
    #     'trt_max_workspace_size': 2147483648, # Set GPU memory usage limit (2GB)
    #     'trt_fp16_enable': True,              # Enable FP16 precision for faster inference  
    # }),
    ('CUDAExecutionProvider', {
        'device_id': 0,                                    # Use first GPU device
        'arena_extend_strategy': 'kNextPowerOfTwo',       # Memory allocation strategy
        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,         # 2GB GPU memory limit
        'cudnn_conv_algo_search': 'EXHAUSTIVE',           # Optimal convolution algorithms
        'do_copy_in_default_stream': True,                # Enable default stream copying
    })
]
def get_config(config_path: str) -> ModuleType:
    """Load and parse a PoseC3D configuration file.
    
    Args:
        config_path: Path to the Python configuration file containing
            transformation pipeline, model paths, and other settings
            
    Returns:
        ModuleType: Loaded configuration module with all settings accessible
            as attributes (e.g., config.transformations, config.ACTION_MODEL_PATH)
    """
    parser = Posec3dConfigParser()
    return parser.parse(config_path)


def get_transformations_pipeline(config: ModuleType) -> IComposer:
    """Create a transformation pipeline from configuration.
    
    Builds a Posec3dComposer with the transformation sequence defined in the
    configuration file. The pipeline handles pose data preprocessing including
    temporal sampling, normalization, and format conversion.
    
    Args:
        config: Configuration module containing 'transformations' attribute
            with a list of transformation dictionaries
            
    Returns:
        IComposer: Configured transformation pipeline composer
    """
    transforms_pipeline = Posec3dComposer(config.transformations)
    return transforms_pipeline


def get_preprocessor(transforms_pipeline: IComposer) -> BasePreprocessor:
    """Create a PoseC3D data preprocessor.
    
    Args:
        transforms_pipeline: Transformation pipeline composer for data processing
        
    Returns:
        BasePreprocessor: Configured Posec3dPreprocessor instance ready for
            processing pose detection results into model input format
    """
    preprocessor = Posec3dPreprocessor(transforms_pipeline)
    return preprocessor


def get_model_engine(config: ModuleType) -> ort.InferenceSession:
    """Create ONNX Runtime inference session for PoseC3D action recognition.
    
    Initializes an ONNX Runtime session with optimized settings for NVIDIA Jetson
    platforms. Configures GPU acceleration, memory limits, and threading for
    best performance on edge devices.
    
    Args:
        config: Configuration module containing 'ACTION_MODEL_PATH' attribute
            with path to the ONNX model file
            
    Returns:
        ort.InferenceSession: Configured ONNX Runtime session ready for inference
        
    Raises:
        Exception: If model loading fails (file not found, invalid format, etc.)
    """
    try:
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1
        model_engine = ort.InferenceSession(
            config.ACTION_MODEL_PATH,session_options=sess_opts, providers=providers
        )
        logger.info(f"Loaded ONNX model: {config.ACTION_MODEL_PATH}")
        return model_engine
    except Exception as e:
        logger.error(f"Failed to load ONNX model {config.ACTION_MODEL_PATH}: {e}")
        raise


def get_yolo_pose_estimator(config: ModuleType) -> YoloPoseEstimator:
    """Create YOLOv8-pose estimator with ONNX Runtime backend.
    
    Initializes a YoloPoseEstimator with optimized ONNX Runtime session for
    pose detection. Configures GPU acceleration and threading for optimal
    performance on NVIDIA Jetson platforms.
    
    Args:
        config: Configuration module containing:
            - YOLO_MODEL: Path to YOLOv8-pose ONNX model file
            - KEYPOINTS_CONF_THRESHOLD: Minimum confidence for keypoint visibility
            
    Returns:
        YoloPoseEstimator: Configured pose estimator ready for inference
        
    Raises:
        Exception: If YOLO model loading fails (file not found, invalid format, etc.)
    """
    try:
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1
        yolo_engine = ort.InferenceSession(
            config.YOLO_MODEL,session_options=sess_opts, providers=providers
        )
        pose_estimator = YoloPoseEstimator(yolo_engine, config.KEYPOINTS_CONF_THRESHOLD)
        logger.info(f"Loaded YOLO model: {config.YOLO_MODEL}")
        return pose_estimator
    except Exception as e:
        logger.error(f"Failed to load YOLO model {config.YOLO_MODEL}: {e}")
        raise


def get_recognizer(
    preprocessor: BasePreprocessor,
    model_engine: ort.InferenceSession,
    label_map_path: str,
) -> IRecognizer:
    """Create PoseC3D action recognition model.
    
    Assembles a complete action recognition pipeline with preprocessor,
    ONNX model engine, and action class labels for classification.
    
    Args:
        preprocessor: Configured data preprocessor for pose sequence processing
        model_engine: ONNX Runtime session for PoseC3D inference
        label_map_path: Path to text file containing action class names
        
    Returns:
        IRecognizer: Complete action recognition model ready for inference
    """
    recognizer = Posec3DRecognizer(preprocessor, model_engine, label_map_path)
    return recognizer


def get_feature_extractor(
    preprocessor: BasePreprocessor, model_engine: ort.InferenceSession
) -> IRecognizer:
    """Create PoseC3D feature extraction model.
    
    Assembles a feature extraction pipeline that returns pose sequence
    embeddings instead of action classifications. Useful for similarity
    matching, clustering, or building custom classifiers.
    
    Args:
        preprocessor: Configured data preprocessor for pose sequence processing
        model_engine: ONNX Runtime session for PoseC3D feature extraction
        
    Returns:
        IRecognizer: Feature extraction model that returns embedding vectors
    """
    extractor = Posec3DFeatureExtractor(preprocessor, model_engine)
    return extractor
