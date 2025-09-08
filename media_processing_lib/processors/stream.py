from media_processing_lib.processors.base import IMediaProcessor
from media_processing_lib.data.schemas import Clip

import gi
import numpy as np
import cv2
import logging
import tempfile
import os
import os.path as osp
from queue import Queue
from collections import deque
from threading import Thread, Event

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

Gst.init(None)

logger = logging.getLogger(__name__)


class StreamProcessor(IMediaProcessor):
    """
    RTSP stream processor with NVIDIA hardware decoding
    """

    def __init__(self, rtsp_url: str, stream_codec: str, rtsp_clips_length: int):
        self.rtsp_url = rtsp_url
        self.clips_length = clips_length

        # Stream properties
        self.stream_codec = stream_codec
        self.fps = None
        self.video_width = None
        self.video_height = None
        self.clip_length_in_frames = None
        self.DEPAY_MAP = {"h264": "rtph264depay", "h265": "rtph265depay"}
        self.PARSER_MAP = {"h264": "h264parse", "h265": "h265parse"}

        # Temp directory management
        self.temp_dir = None
        self.target_dir = None
        self.frame_counter = 0

        # GStreamer pipeline
        self.pipeline = None
        self.app_sink = None
        self.main_loop = None
        self.stop_event = Event()

        # Frame processing
        self.clip_container = None

    def _create_pipeline(self):
        """Create GStreamer pipeline with NVIDIA hardware decoding"""

        # Create pipeline string for RTSP with NVIDIA decoding. nvv4l2decoder -> should be the decoder for jetson.
        pipeline_str = f"""
        rtspsrc location={self.rtsp_url} latency=0 ! 
        {self.DEPAY_MAP[self.stream_codec]} ! 
        {self.PARSER_MAP[self.stream_codec]} ! 
        nvv4l2decoder enable-max-performance=1 ! 
        nvvideoconvert ! 
        video/x-raw,format=BGR ! 
        appsink name=sink emit-signals=true sync=false max-buffers=2 drop=true
        """

        logger.info(f"Creating pipeline: {pipeline_str}")
        self.pipeline = Gst.parse_launch(pipeline_str)

        # Get app sink
        self.app_sink = self.pipeline.get_by_name("sink")
        if not self.app_sink:
            raise RuntimeError("Failed to get appsink element")

        # Configure app sink
        self.app_sink.set_property("emit-signals", True)
        self.app_sink.set_property("sync", False)
        self.app_sink.set_property("max-buffers", 2)
        self.app_sink.set_property("drop", True)

        # Connect callback
        self.app_sink.connect("new-sample", self._on_new_sample)

    def _on_new_sample(self, app_sink):
        """Callback for new frame from appsink"""
        try:
            sample = app_sink.emit("pull-sample")
            if not sample:
                return Gst.FlowReturn.OK

            buffer = sample.get_buffer()
            caps = sample.get_caps()

            # Get frame dimensions
            structure = caps.get_structure(0)
            width = structure.get_int("width")[1]
            height = structure.get_int("height")[1]

            # Extract frame data
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if not success:
                logger.error("Failed to map buffer")
                return Gst.FlowReturn.OK

            # Convert buffer data to numpy array (should be good to go since we have the nvvideoconvert before the appsink)
            frame_data = np.ndarray(
                shape=(height, width, 3), buffer=map_info.data, dtype=np.uint8
            )
            frame = np.copy(frame_data)
            buffer.unmap(map_info)

            self._process_frame(frame)

            return Gst.FlowReturn.OK

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return Gst.FlowReturn.ERROR

    def _process_frame(self, frame):
        """Process individual frame - save to disk and manage clips"""
        if self.stop_event.is_set():
            return

        # Save frame to disk
        frame_tmpl = osp.join(self.target_dir, "img_{:06d}.jpg")
        frame_path = frame_tmpl.format(self.frame_counter + 1)

        success = cv2.imwrite(frame_path, frame)
        if not success:
            logger.error(f"Failed to save frame to {frame_path}")
            return

        self.clip_container.append(frame_path)
        self.frame_counter += 1

        # Create clip when buffer is full
        if len(self.clip_container) == self.clip_length_in_frames:
            current_clip_frames = list(self.clip_container)
            clip = Clip(
                frame_paths=current_clip_frames,
                clip_size=len(current_clip_frames),
                temp_dir=self.temp_dir,
            )
            self.clips_queue.put(clip)
            logger.debug(f"Created clip with {len(current_clip_frames)} frames")
            self.clip_container.clear()

    def init_frame_repo(self):
        """Initialize temporary directory for frame storage"""
        self.temp_dir = tempfile.mkdtemp(prefix="stream_frames_")
        stream_name = self.rtsp_url.split("/")[-1] or "stream"
        self.target_dir = osp.join(self.temp_dir, stream_name)
        os.makedirs(self.target_dir, exist_ok=True)
        logger.info(f"Initialized frame repository: {self.target_dir}")

    def configure(self):
        """Configure stream processor"""
        # For streams, we estimate these values or get them from caps
        self.fps = 30  # Default
        self.clip_length_in_frames = int(self.fps * self.clips_length)
        self.clip_container = deque(maxlen=self.clip_length_in_frames)

        self.init_frame_repo()
        self._create_pipeline()

    def start(self, clips_queue: Queue):
        """Start stream processing"""
        self.clips_queue = clips_queue
        try:
            logger.info(f"Starting RTSP stream: {self.rtsp_url}")

            # Set pipeline to playing
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("Failed to start pipeline")

            # Create and run main loop -> this is the loop that lets us comunicate with the gstreamer-processing thread.
            self.main_loop = GLib.MainLoop()

            # Run main loop in separate thread to avoid blocking
            loop_thread = Thread(target=self.main_loop.run, daemon=True)
            loop_thread.start()

            # Wait for stop signal
            self.stop_event.wait()

        except Exception as e:
            logger.error(f"Error in stream processing: {e}")
        finally:
            self._cleanup_pipeline()

    def stop(self):
        """Stop stream processing"""
        logger.info("Stopping stream processor")
        self.stop_event.set()

    def _cleanup_pipeline(self):
        """Clean up GStreamer pipeline"""
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline = None

        if self.main_loop:
            self.main_loop.quit()
            self.main_loop = None

    def cleanup(self):
        """Clean up temporary directory"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil

            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temp directory: {self.temp_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    rtsp_url = (
        "rtsp://vtviewer:Vtech123!@192.168.1.120/cam/realmonitor?channel=1&subtype=2"
    )
    processor = StreamProcessor(rtsp_url, clips_length=2)

    try:
        processor.configure()
        clips_queue = Queue()

        stream_thread = Thread(target=processor.start, args=(clips_queue,))
        stream_thread.start()

        while True:
            try:
                clip = clips_queue.get(timeout=5)
                logger.info(f"Got clip with {clip.clip_size} frames")
            except:
                logger.info("No clips received")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        processor.stop()
        processor.cleanup()
