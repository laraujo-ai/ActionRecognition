from media_processing_lib.processors.video import VideoProcessor
from media_processing_lib.processors.stream import StreamProcessor
from media_processing_lib.processors.base import IMediaProcessor


def get_video_processor(media_link: str, clip_length: int) -> IMediaProcessor:
    """Create a video file processor for local video files.

    Factory function that creates a VideoProcessor instance configured
    for processing local video files (MP4, AVI, MOV, etc.).

    Args:
        media_link: Path to local video file
        clip_length: Duration of each clip in seconds

    Returns:
        IMediaProcessor: Configured video processor instance

    Example:
        ```python
        processor = get_video_processor("./video.mp4", clip_length=2)
        processor.configure()
        processor.start(clips_queue)
        ```
    """
    return VideoProcessor(media_link, clip_length)


def get_stream_processor(
    media_link: str, stream_codec: str, clip_length: int, fps :int
) -> IMediaProcessor:
    """Create a stream processor for RTSP/live video streams.

    Factory function that creates a StreamProcessor instance configured
    for processing live video streams with hardware-accelerated decoding.

    Args:
        media_link: RTSP stream URL (e.g., "rtsp://192.168.1.100:554/stream")
        stream_codec: Video codec for stream processing (e.g., "h264", "h265")
        clip_length: Duration of each clip in seconds
        fps : The desired rate for clip recording
    Returns:
        IMediaProcessor: Configured stream processor instance

    Example:
        ```python
        processor = get_stream_processor(
            "rtsp://camera.local/stream",
            "h264",
            clip_length=2
        )
        processor.configure()
        processor.start(clips_queue)
        ```
    """
    return StreamProcessor(media_link, stream_codec, clip_length, fps)
