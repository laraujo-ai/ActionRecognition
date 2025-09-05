import cv2
import numpy as np
import argparse
import tempfile
import logging

from utils.drawing import draw_on_video
from utils.video_recording import extract_frames
import providers as prov


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("posec3d.log"), logging.StreamHandler()],
    )


def parse_args():
    parser = argparse.ArgumentParser("VisionTech Action Recognition Demo")
    parser.add_argument("--video_path", type=str)
    parser.add_argument("--config_path", type=str)
    args = parser.parse_args()

    return args


def main(args):
    logger = logging.getLogger(__name__)

    try:
        logger.info("Starting action recognition demo")
        config = prov.get_config(args.config_path)
        pose_estimator = prov.get_yolo_pose_estimator(config)

        transforms_pipeline = prov.get_transformations_pipeline(config)
        preprocessor = prov.get_preprocessor(transforms_pipeline)
        onnx_model_engine = prov.get_model_engine(config)

        recognizer = prov.get_recognizer(
            preprocessor, onnx_model_engine, config.LABEL_MAP_PATH
        )

        logger.info(f"Processing video: {args.video_path}")
        tmp_dir = tempfile.TemporaryDirectory()
        frame_paths, frames, fps = extract_frames(args.video_path, tmp_dir.name)
        h, w, _ = frames[0].shape

        logger.info(f"Extracted {len(frames)} frames at {fps} fps")
        pose_results = pose_estimator.inference_on_video(frame_paths)
        data = {"pose_results": pose_results, "img_shape": (h, w)}

        predicted_action = recognizer.inference(data)
        logger.info(f"Predicted action: {predicted_action}")

        frames = draw_on_video(frames, pose_results, predicted_action)

        delay_ms = int(round(1000 / fps))  # 33
        cv2.namedWindow("vis", 0)
        for frame in frames:
            cv2.imshow("vis", frame)
            if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                break

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    main(args)
