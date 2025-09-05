from types import ModuleType
import logging
import onnxruntime as ort
from ultralytics import YOLO

from utils.composer import Posec3dComposer, IComposer
from models.recognizers.Posec3D import (
    Posec3DRecognizer,
    IRecognizer,
    Posec3DFeatureExtractor,
)
from utils.parser import Posec3dConfigParser
from models.preprocessors.posec3d_preprocessor import (
    BasePreprocessor,
    Posec3dPreprocessor,
)
from models.pose_estimators.yolo_estimator import YoloPoseEstimator

logger = logging.getLogger(__name__)


def get_config(config_path: str) -> ModuleType:
    parser = Posec3dConfigParser()
    return parser.parse(config_path)


def get_transformations_pipeline(config: ModuleType) -> IComposer:
    transforms_pipeline = Posec3dComposer(config.transformations)
    return transforms_pipeline


def get_preprocessor(transforms_pipeline: IComposer) -> BasePreprocessor:
    preprocessor = Posec3dPreprocessor(transforms_pipeline)
    return preprocessor


def get_model_engine(config: ModuleType) -> ort.InferenceSession:
    try:
        model_engine = ort.InferenceSession(
            config.ACTION_MODEL_PATH, providers=["CPUExecutionProvider"]
        )
        logger.info(f"Loaded ONNX model: {config.ACTION_MODEL_PATH}")
        return model_engine
    except Exception as e:
        logger.error(f"Failed to load ONNX model {config.ACTION_MODEL_PATH}: {e}")
        raise


def get_yolo_pose_estimator(config: ModuleType) -> YoloPoseEstimator:
    try:
        yolo_engine = YOLO(config.YOLO_MODEL)
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
    recognizer = Posec3DRecognizer(preprocessor, model_engine, label_map_path)
    return recognizer


def get_feature_extractor(
    preprocessor: BasePreprocessor, model_engine: ort.InferenceSession
) -> IRecognizer:
    extractor = Posec3DFeatureExtractor(preprocessor, model_engine)
    return extractor
