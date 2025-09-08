import numpy as np
import logging
import gc


logger = logging.getLogger(__name__)


def inference_thread(pose_estimator, action_model, results_queue, clips_queue) -> None:

    while True:
        try:
            clip = clips_queue.get(timeout=1)
            if clip is None:  # Poison pill to stop thread
                break

            if clip.clip_size <= 0:
                continue

            pose_results = pose_estimator.inference_on_video(clip.frame_paths)
            h, w = clip.frame_shape

            data = {"pose_results": pose_results, "img_shape": (h, w)}
            inference_results = action_model.inference(data)

            logger.info(f"Clip processed: {inference_results}")

            assert isinstance(inference_results, str) or isinstance(
                inference_results, np.ndarray
            )

            results_queue.put(inference_results)
            logger.info(f"Clip processed: {inference_results}")

            # Memory cleanup
            del clip, pose_results, data
            gc.collect()

        except Exception as e:
            from queue import Empty as QueueEmpty

            if isinstance(e, QueueEmpty):
                logger.debug("Queue timeout, continuing...")
                continue
            else:
                logger.error(f"Inference error: {e}")
                continue
