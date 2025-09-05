from typing import Protocol, Dict, Any
import importlib.util
import os
from types import ModuleType


class BaseParser(Protocol):
    def __init__(self): ...
    def parse(self, file_path: str) -> Any: ...


class Posec3dConfigParser(BaseParser):
    def __init__(self): ...

    def parse(self, file_path: str) -> ModuleType:
        config_module = self.import_module_from_path(file_path)
        return config_module

    def import_module_from_path(self, file_path: str) -> ModuleType:
        file_path = os.path.abspath(file_path)
        spec = importlib.util.spec_from_file_location("config_module", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # execute the module
        return module
