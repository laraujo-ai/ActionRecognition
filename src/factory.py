import argparse
import logging

import media_processing_lib.providers as media_prov
import posec3d_lib.providers as posec3d_prov
from src.utils import is_rtsp_stream


def create_action_recognizer(model_config):
    """Create PoseC3D action recognition model with preprocessing pipeline.

    Args:
        model_config: Model configuration containing paths and parameters

    Returns:
        IRecognizer: Configured action recognition model
    """
    transforms_pipeline = posec3d_prov.get_transformations_pipeline(model_config)
    preprocessor = posec3d_prov.get_preprocessor(transforms_pipeline)
    onnx_model_engine = posec3d_prov.get_model_engine(model_config)
    return posec3d_prov.get_recognizer(
        preprocessor, onnx_model_engine, model_config.LABEL_MAP_PATH
    )


def create_feature_extractor(model_config):
    """Create PoseC3D feature extraction model with preprocessing pipeline.

    Args:
        model_config: Model configuration containing paths and parameters

    Returns:
        IRecognizer: Configured feature extraction model
    """
    transforms_pipeline = posec3d_prov.get_transformations_pipeline(model_config)
    preprocessor = posec3d_prov.get_preprocessor(transforms_pipeline)
    onnx_model_engine = posec3d_prov.get_model_engine(model_config)
    return posec3d_prov.get_feature_extractor(preprocessor, onnx_model_engine)


def create_models(args: argparse.Namespace, logger: logging.Logger) -> tuple:
    """Create pose estimator and action recognition models based on specified type.

    Args:
        args: Command line arguments containing model type
        logger: Logger instance for status messages

    Returns:
        tuple: (pose_estimator, model) or (None, None) if model type unsupported

    Raises:
        RuntimeError: If model loading fails
    """
    try:
        if args.model_type == "classifier":
            logger.info("Loading PoseC3D classifier")
            model_config = posec3d_prov.get_config(
                "posec3d_lib/configs/posec3d_inference.py"
            )
            pose_estimator = posec3d_prov.get_yolo_pose_estimator(model_config)
            model = create_action_recognizer(model_config)
            return pose_estimator, model

        elif args.model_type == "feature_extractor":
            logger.info("Loading PoseC3D Feature Extractor")
            model_config = posec3d_prov.get_config(
                "posec3d_lib/configs/posec3d_feature_extractor.py"
            )
            pose_estimator = posec3d_prov.get_yolo_pose_estimator(model_config)
            model = create_feature_extractor(model_config)
            return pose_estimator, model

        else:
            logger.error(f"Unsupported model type: {args.model_type}")
            return None, None

    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise RuntimeError(f"Model loading failed: {e}")


def create_media_processor(args: argparse.Namespace):
    """Create appropriate media processor based on input type.

    Args:
        args: Command line arguments containing media configuration

    Returns:
        IMediaProcessor: Stream processor for RTSP or video processor for files
    """
    if is_rtsp_stream(args.media_link):
        return media_prov.get_stream_processor(
            args.media_link, args.stream_codec, args.clips_length
        )
    else:
        return media_prov.get_video_processor(args.media_link, args.clips_length)
