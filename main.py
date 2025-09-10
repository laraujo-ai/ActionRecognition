import argparse
import logging
from queue import Queue, Empty
from threading import Thread

from src.inference import inference_thread
from src.factory import create_models, create_media_processor, create_anomaly_detector


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for action recognition system.

    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - media_link: Path to video file or RTSP stream URL
            - model_type: Type of model ('classifier' or 'anomaly_detector')
            - stream_codec: Codec for stream processing
            - clips_length: Length of video clips in seconds
    """
    parser = argparse.ArgumentParser("VisionTech Action Recognition POC")
    parser.add_argument(
        "--media_link",
        type=str,
        required=True,
        help="Path to video file or RTSP stream URL",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["classifier", "anomaly_detector"],
        help="Type of model to use",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        help="Path to your pre-trained mahalanobis anomaly detection model",
    )
    parser.add_argument(
        "--stream_codec", type=str, default="h264", help="Codec for stream processing"
    )
    parser.add_argument(
        "--clips_length", type=int, default=2, help="Length of video clips in seconds"
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="The desired Fps for inference"
    )
    return parser.parse_args()


def setup_logging() -> None:
    """Configure logging for the application with INFO level and timestamp format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def main(args: argparse.Namespace) -> None:
    """Main application entry point for action recognition system.

    Args:
        args: Parsed command line arguments
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info(f"Starting action recognition with media: {args.media_link}")

    clips_queue = Queue(maxsize=5)
    results_queue = Queue(maxsize=10)

    try:
        pose_estimator, model = create_models(args, logger)
        anomaly_detector = None
        if not pose_estimator or not model:
            logger.error("Failed to load required models")
            return

        if args.model_type == "anomaly_detector":
            anomaly_detector = create_anomaly_detector(args.pretrained_path)

        media_processor = create_media_processor(args)
        media_processor.configure()

        video_thread = Thread(
            target=media_processor.start, args=(clips_queue,), daemon=True
        )
        model_thread = Thread(
            target=inference_thread,
            args=(pose_estimator, model, anomaly_detector, results_queue, clips_queue),
            daemon=True,
        )

        video_thread.start()
        model_thread.start()
        logger.info("Processing threads started")

        while True:
            try:
                result = results_queue.get(timeout=5)
                logger.info(f"Output : {result}")
            except Empty:
                if not video_thread.is_alive() and not model_thread.is_alive():
                    logger.info("All threads finished")
                    break

    except KeyboardInterrupt:
        logger.exception("Interrupted by user")
        clips_queue.put(None)
    except Exception as e:
        logger.exception("Application error")
    finally:
        if "media_processor" in locals() and hasattr(media_processor, "cleanup"):
            media_processor.cleanup()


if __name__ == "__main__":
    args = parse_args()
    main(args)
