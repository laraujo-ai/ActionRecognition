from typing import Protocol
from queue import Queue


class IMediaProcessor(Protocol):
    """Protocol interface for media processing components.

    Defines the contract that all media processors (video files, RTSP streams, etc.)
    must implement to participate in the action recognition pipeline.

    The typical workflow is:
    1. Initialize processor with media source and configuration
    2. Call configure() to set up processing parameters
    3. Call start() to begin processing and producing clips
    4. Call cleanup() when processing is complete
    """

    def __init__(self, **kwargs) -> None:
        """Initialize media processor with source-specific parameters.

        Args:
            **kwargs: Source-specific configuration parameters
        """
        ...

    def configure(self) -> None:
        """Configure processor settings and initialize resources.

        Sets up frame repositories, validates media source, calculates
        processing parameters like frame rates and clip sizes.
        """
        ...

    def start(self, clips_queue: Queue) -> None:
        """Begin media processing and produce clips.

        Processes the media source and produces Clip objects containing
        frame paths and metadata. Clips are pushed to the provided queue
        for downstream consumption.

        Args:
            clips_queue: Queue to receive produced Clip objects
        """
        ...

    def cleanup(self) -> None:
        """Clean up resources and temporary files.

        Removes temporary directories, closes file handles, and performs
        any other necessary cleanup operations.
        """
        ...

    def init_frame_repo(self) -> None:
        """Initialize temporary storage for frame files.

        Creates temporary directories and sets up file naming conventions
        for storing individual video frames during processing.
        """
        ...
