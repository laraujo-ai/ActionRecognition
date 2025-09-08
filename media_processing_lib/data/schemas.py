from pydantic import BaseModel
from typing import Protocol, Sequence


class Clip(BaseModel):
    frame_paths: Sequence[str]
    frame_shape: tuple
    clip_size: int
    temp_dir: str
