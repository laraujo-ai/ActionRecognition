from typing import Protocol, Sequence, Any
from utils.composer import IComposer


class BasePreprocessor(Protocol):
    def __init__(self, transforms_pipeline) : ...
    def process(self, data : Any) -> Any : ...

