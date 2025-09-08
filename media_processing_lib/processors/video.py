import cv2
import os
import os.path as osp
import tempfile
from collections import deque
from queue import Queue
import threading
import logging
from media_processing_lib.processors.base import IMediaProcessor
from media_processing_lib.data.schemas import Clip

logger = logging.getLogger(__name__)


class VideoProcessor(IMediaProcessor):
    """
    The clips_length here are in seconds
    """

    def __init__(self, media_link: str, clips_length: int):
        self.media_link = media_link
        self.cap = cv2.VideoCapture(media_link)
        self.clips_length = clips_length

        if not self.cap.isOpened():
            raise ValueError(f"Error opening video: {media_link}")

        self.fps = None
        self.video_width: int = None
        self.video_height: int = None
        self.clip_length_in_frames: int = None
        self.total_frames: int = None

        self.clip_container = None
        self.temp_dir = None
        self.target_dir = None
        self.frame_counter = 0
        self.num_clips = 0

    def init_frame_repo(self):
        self.temp_dir = tempfile.mkdtemp(prefix="video_frames_")
        video_name = osp.basename(osp.splitext(self.media_link)[0])
        self.target_dir = osp.join(self.temp_dir, video_name)
        os.makedirs(self.target_dir, exist_ok=True)
        logger.info(f"Initialized frame repository: {self.target_dir}")

    def configure(self):
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.clip_length_in_frames = int(self.fps * self.clips_length)

        if self.clip_length_in_frames > self.total_frames:
            logger.warning(
                f"Clip length ({self.clip_length_in_frames}) > total frames ({self.total_frames}). Using total video size for clip length ..."
            )
            self.clip_length_in_frames = self.total_frames

        self.num_clips = self.total_frames // self.clip_length_in_frames
        self.clip_container = deque(maxlen=self.clip_length_in_frames)
        self.init_frame_repo()

    def cleanup(self):
        """Clean up temporary directory when done"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil

            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temp directory: {self.temp_dir}")

    def start(self, clips_queue: Queue):
        try:
            frame_tmpl = osp.join(self.target_dir, "img_{:06d}.jpg")
            clip_number = 0
            while self.frame_counter < self.total_frames:
                current_clip_frames = []
                for _ in range(self.clip_length_in_frames):
                    if self.frame_counter >= self.total_frames:
                        break

                    ret, frame = self.cap.read()
                    if not ret:
                        logger.warning(f"Failed to read frame {self.frame_counter}")
                        break

                    frame_path = frame_tmpl.format(self.frame_counter + 1)
                    success = cv2.imwrite(frame_path, frame)
                    if not success:
                        logger.error(f"Failed to save frame to {frame_path}")
                        continue

                    current_clip_frames.append(frame_path)
                    self.frame_counter += 1

                # Create clip if we have enough frames
                if len(current_clip_frames) >= self.clip_length_in_frames:
                    frame_shape = (self.video_height, self.video_width)
                    clip = Clip(
                        frame_paths=current_clip_frames,
                        clip_size=len(current_clip_frames),
                        temp_dir=self.temp_dir,
                        frame_shape=frame_shape,
                    )
                    clips_queue.put(clip)
                    clip_number += 1
                    logger.info(
                        f"Created clip {clip_number} with {len(current_clip_frames)} frames"
                    )

        except Exception as e:
            logger.error(f"Error in video processing: {e}")
        finally:
            self.cap.release()
            logger.info(
                f"Video processing completed. Created {clip_number} clips from {self.frame_counter} frames"
            )


if __name__ == "__main__":
    video_capturer = VideoProcessor("./falling3.mp4", 2)
    video_capturer.configure()

    clips_queue = Queue()
    video_processor_thread = threading.Thread(
        target=video_capturer.start, args=(clips_queue,)
    )
    video_processor_thread.start()

    try:
        clip = clips_queue.get()
        print(f"Got clip alright : {clip.clip_size}")
    except Exception as e:
        print(f"An error ocurred : {e}")
