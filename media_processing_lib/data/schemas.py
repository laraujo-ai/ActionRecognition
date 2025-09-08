from pydantic import BaseModel, Field
from typing import Sequence, Tuple


class Clip(BaseModel):
    """Represents a processed video clip with frame metadata.

    A clip contains references to individual frame files on disk along with
    metadata about the video segment. Frames are stored as individual image
    files in a temporary directory to optimize memory usage.

    Attributes:
        frame_paths: Ordered sequence of absolute paths to individual frame files
        frame_shape: Video frame dimensions as (height, width) tuple
        clip_size: Number of frames in this clip
        temp_dir: Path to temporary directory containing frame files

    Example:
        ```python
        clip = Clip(
            frame_paths=["/tmp/frames/img_000001.jpg", "/tmp/frames/img_000002.jpg"],
            frame_shape=(1080, 1920),
            clip_size=60,
            temp_dir="/tmp/video_frames_abc123"
        )
        ```
    """

    frame_paths: Sequence[str] = Field(
        ..., description="Paths to individual frame image files"
    )
    frame_shape: Tuple[int, int] = Field(
        ..., description="Frame dimensions (height, width)"
    )
    clip_size: int = Field(..., description="Number of frames in clip", gt=0)
    temp_dir: str = Field(..., description="Temporary directory containing frame files")
