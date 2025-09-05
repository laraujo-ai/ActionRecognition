from typing import Protocol, Any, Dict, List, Optional, Sequence
import importlib
from functools import partial


def import_from_string(path: str):
    """Resolve 'pkg.module.attr' -> the attribute (function/class)."""
    module_name, _, attr = path.rpartition(".")
    if not module_name:
        raise ValueError(f"import path must be 'module.attr', got {path}")
    mod = importlib.import_module(module_name)
    return getattr(mod, attr)


class IComposer(Protocol):
    def __init__(self, transformations): ...
    def __call__(self, data: Any) -> Any: ...


class Posec3dComposer(IComposer):
    """
    Compose transforms. Accepts:
      - dicts: {"type": "package.module.ClsOrFn", ...kwargs}
    """

    def __init__(self, transforms: Optional[Sequence[Dict]] = None):
        self.transforms: List[Callable] = []
        transforms = transforms or []
        for t in transforms:
            assert isinstance(t, dict)
            cfg = t.copy()
            if "type" not in cfg:
                raise ValueError("dict transform must include 'type' key")
            typ = cfg.pop("type")
            obj = import_from_string(typ)
            if isinstance(obj, type):
                inst = obj(
                    **cfg
                )  # in this case the obj is a class, then we just unwrap the cfg args into the constructor of the class
            else:
                # wrap function with provided kwargs
                inst = partial(obj, **cfg)
            self.transforms.append(inst)

    def __call__(self, data: dict) -> Optional[dict]:
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data
