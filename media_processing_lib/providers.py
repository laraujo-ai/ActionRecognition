from media_processing_lib.processors.video import VideoProcessor
from media_processing_lib.processors.stream import StreamProcessor
from media_processing_lib.processors.base import IMediaProcessor


def get_stream_processor(): ...


def get_video_processor(media_link: str, clip_length: int) -> IMediaProcessor:
    processor = VideoProcessor(media_link, clip_length)
    return processor


def get_stream_processor(
    media_link: str, stream_codec: str, clip_length
) -> IMediaProcessor:
    processor = StreamProcessor(media_link, stream_codec, clip_length)
    return processor
