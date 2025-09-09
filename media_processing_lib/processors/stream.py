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
from typing import Optional, Deque

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

Gst.init(None)

logger = logging.getLogger(__name__)


class StreamProcessor(IMediaProcessor):
    """RTSP stream processor with NVIDIA hardware-accelerated decoding.

    Processes live RTSP video streams using GStreamer pipeline with NVIDIA GPU
    acceleration for H.264/H.265 decoding. Continuously captures frames and
    creates non-overlapping video clips for action recognition processing.

    The processing pipeline:
    RTSP Source → RTP Depay → Parser → NV Decoder → Color Convert → App Sink

    Attributes:
        rtsp_url: RTSP stream URL to process
        clips_length: Duration of each clip in seconds
        stream_codec: Video codec type ("h264" or "h265")

    Example:
        ```python
        processor = StreamProcessor(
            rtsp_url="rtsp://192.168.1.100:554/stream",
            clips_length=2
        )
        processor.configure()
        processor.start(clips_queue)
        ```
    """

    def __init__(self, rtsp_url: str, stream_codec : str ,clips_length: int, fps:int) -> None:
        """Initialize RTSP stream processor.

        Args:
            rtsp_url: RTSP stream URL to connect to
            stream_codec : The codec in which the stream is embedded
            clips_length: Duration of each video clip in seconds

        Raises:
            ValueError: If RTSP URL is invalid
        """
        self.rtsp_url = rtsp_url
        self.clips_length = clips_length
        self.stream_codec = stream_codec
        self.fps = fps

        self.video_width = None
        self.video_height = None
        self.clip_length_in_frames = None
        self.DEPAY_MAP = {"h264": "rtph264depay", "h265": "rtph265depay"}
        self.PARSER_MAP = {"h264": "h264parse", "h265": "h265parse"}

        self.temp_dir = None
        self.target_dir = None
        self.frame_counter = 0

        self.pipeline = None
        self.app_sink = None
        self.main_loop = None
        self.stop_event = Event()

        self.width = None
        self.height = None

        self.clip_container: Optional[Deque[str]] = None

    def _create_pipeline(self) -> None:
        """Create GStreamer pipeline with NVIDIA hardware decoding.

        Builds a GStreamer pipeline optimized for RTSP streams with hardware
        acceleration using NVIDIA decoders. The pipeline automatically handles
        RTP depayloading, parsing, and color space conversion.

        Pipeline structure:
        rtspsrc -> rtpXXXdepay -> XXXparse -> nvv4l2decoder -> nvvideoconvert -> appsink

        Raises:
            RuntimeError: If pipeline creation or element linking fails
        """

        # Create pipeline string for RTSP with NVIDIA decoding. The hardware decoder gives us back a NV12 format frame. 
        # Unfortunately the VIC(Vision Image Compositor) software, which is what the nvvideoconvert is trying to use to convert our frame
        # to RGB does not suport this convertion operation. Therefore we include a basic videoconvert after the nvvideoconvert.
        # Now the nvvideoconvert just takes the NV12 buffer and transfers it to the CPU memory and the videoconvert gets us the RGB frame.
        
        pipeline_str = f"""
        rtspsrc location="{self.rtsp_url}" latency=0 protocol=tcp !
        {self.DEPAY_MAP[self.stream_codec]} !
        {self.PARSER_MAP[self.stream_codec]} !
        nvv4l2decoder enable-max-performance=1 !
        nvvideoconvert !
        videorate !
        video/x-raw,width=640,height=640,framerate={self.fps}/1 !
        videoconvert !
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
        
        # Add bus message handler for error monitoring
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::error", self._on_pipeline_error)
        bus.connect("message::warning", self._on_pipeline_warning)
        bus.connect("message::info", self._on_pipeline_info)

    def _cleanup_pipeline(self):
        """Clean up GStreamer pipeline"""
        if self.pipeline:
            bus = self.pipeline.get_bus()
            if bus:
                bus.remove_signal_watch()
            
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline = None    

        if self.main_loop:
            self.main_loop.quit()
            self.main_loop = None

    def _on_pipeline_error(self, bus, msg):
        """Handle pipeline error messages"""
        err, debug = msg.parse_error()
        logger.error(f"Pipeline error: {err} - Debug: {debug}")
        self.stop_event.set()
        
    def _on_pipeline_warning(self, bus, msg):
        """Handle pipeline warning messages"""
        warn, debug = msg.parse_warning()
        logger.warning(f"Pipeline warning: {warn} - Debug: {debug}")
        
    def _on_pipeline_info(self, bus, msg):
        """Handle pipeline info messages"""
        info, debug = msg.parse_info()
        logger.info(f"Pipeline info: {info} - Debug: {debug}")

    def _on_new_sample(self, app_sink):
        """Callback for new frame from appsink"""
        sample = None
        try:
            sample = app_sink.emit("pull-sample")
            if not sample:
                return Gst.FlowReturn.OK

            buffer = sample.get_buffer()
            caps = sample.get_caps()

            # Get frame dimensions
            structure = caps.get_structure(0)
            self.width = structure.get_int("width")[1]
            self.height = structure.get_int("height")[1]

            # Extract frame data
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if not success:
                logger.error("Failed to map buffer")
                return Gst.FlowReturn.OK

            try:
                # Convert buffer data to numpy array
                frame_data = np.ndarray(
                    shape=(self.height, self.width, 3), buffer=map_info.data, dtype=np.uint8
                )
                frame = np.copy(frame_data)
                self._process_frame(frame)
            finally:
                buffer.unmap(map_info)
            return Gst.FlowReturn.OK
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return Gst.FlowReturn.ERROR
        finally:
            if sample is not None:
                del sample

    def _process_frame(self, frame):
        """Process individual frame - save to disk and manage clips"""
        if self.stop_event.is_set():
            return

        # Save frame to disk
        if not self.target_dir:
            logger.error("Target directory not initialized")
            return
        frame_tmpl = osp.join(self.target_dir, "img_{:06d}.jpg")
        frame_path = frame_tmpl.format(self.frame_counter + 1)

        success = cv2.imwrite(frame_path, frame)
        if not success:
            logger.error(f"Failed to save frame to {frame_path}")
            return

        if self.clip_container is not None:
            self.clip_container.append(frame_path)
        self.frame_counter += 1

        # Create clip when buffer is full
        if self.clip_container is not None and len(self.clip_container) == self.clip_length_in_frames:
            current_clip_frames = list(self.clip_container)
            clip = Clip(
                frame_paths=current_clip_frames,
                clip_size=len(current_clip_frames),
                temp_dir=self.temp_dir,
                frame_shape=(self.height, self.width)
            )
            self.clips_queue.put(clip)
            logger.debug(f"Created clip with {len(current_clip_frames)} frames")
            if self.clip_container is not None:
                self.clip_container.clear()

    def init_frame_repo(self) -> None:
        """Initialize temporary directory for frame storage.

        Creates a unique temporary directory for storing individual frame files
        captured from the RTSP stream. Uses stream identifier for organization.
        """
        self.temp_dir = tempfile.mkdtemp(prefix="stream_frames_")
        stream_name = self.rtsp_url.split("/")[-1] or "stream"
        self.target_dir = osp.join(self.temp_dir, stream_name)
        os.makedirs(self.target_dir, exist_ok=True)
        logger.info(f"Initialized frame repository: {self.target_dir}")

    def configure(self) -> None:
        """Configure stream processor settings and initialize resources.

        Sets up processing parameters, creates temporary storage, and builds
        the GStreamer pipeline.
        """

        self.clip_length_in_frames = int(self.fps * self.clips_length)
        self.clip_container = deque(maxlen=self.clip_length_in_frames)

        self.init_frame_repo()
        self._create_pipeline()

    def start(self, clips_queue: Queue) -> None:
        """Start RTSP stream processing and clip generation.

        Begins capturing frames from the RTSP stream and produces video clips.
        Runs until stop() is called or an error occurs. The GStreamer main loop
        handles frame callbacks while this thread waits for completion.

        Args:
            clips_queue: Queue to receive generated Clip objects

        Raises:
            RuntimeError: If pipeline fails to start
        """
        self.clips_queue = clips_queue
        try:
            logger.info(f"Starting RTSP stream: {self.rtsp_url}")

            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("Failed to start pipeline")

            self.main_loop = GLib.MainLoop()
            loop_thread = Thread(target=self.main_loop.run, daemon=True)
            loop_thread.start()
            self.stop_event.wait()

        except Exception as e:
            logger.error(f"Error in stream processing: {e}")
        finally:
            self._cleanup_pipeline()

    def stop(self) -> None:
        """Stop stream processing gracefully.

        Signals the processing thread to stop and initiates pipeline shutdown.
        """
        logger.info("Stopping stream processor")
        self.stop_event.set()

    def cleanup(self) -> None:
        """Clean up temporary directory and frame files.

        Removes all temporary frame files and directories created during
        stream processing. Should be called after processing is complete.
        """
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil

            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temp directory: {self.temp_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    rtsp_url = "rtsp//..."
    processor = StreamProcessor(rtsp_url, "h264", clips_length=2)

    try:
        processor.configure()
        clips_queue = Queue()

        stream_thread = Thread(target=processor.start, args=(clips_queue,))
        stream_thread.start()

        while True:
            try:
                clip = clips_queue.get(timeout=5)
                logger.info(f"Got clip with {clip.clip_size} frames")
            except Exception as e:
                logger.info("No clips received")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        processor.stop()
        processor.cleanup()
