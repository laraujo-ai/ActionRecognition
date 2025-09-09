from typing import Protocol, Any, Dict, List, Optional, Sequence, Callable
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
    """Protocol interface for transformation pipeline composers.
    
    Defines the contract that all composers must implement for building
    and executing transformation pipelines on data.
    """
    def __init__(self, transformations):
        """Initialize composer with transformation specifications.
        
        Args:
            transformations: Sequence of transformation configurations
        """
        ...
    def __call__(self, data: Any) -> Any:
        """Apply the transformation pipeline to input data.
        
        Args:
            data: Input data to transform
            
        Returns:
            Any: Transformed data after applying all pipeline steps
        """
        ...


class Posec3dComposer(IComposer):
    """Compose and execute transformation pipelines for PoseC3D data processing.
    
    Builds a sequential pipeline of transformations from configuration dictionaries.
    Each transformation can be either a class (instantiated with provided kwargs)
    or a function (wrapped with partial application of kwargs).
    
    Configuration format:
    - Each transform is a dict with required 'type' key specifying import path
    - Additional keys become kwargs for the class constructor or function
    - Pipeline executes transforms sequentially, passing data between steps
    - If any transform returns None, the pipeline short-circuits and returns None
    
    Example:
        transforms = [
            {"type": "posec3d_lib.data.pose_transforms.uniform_sample_frames", "clip_len": 48},
            {"type": "posec3d_lib.data.processing.center_crop", "crop_size": 64}
        ]
        composer = Posec3dComposer(transforms)
        result = composer(input_data)
    
    Attributes:
        transforms: List of instantiated transformation callables
    """

    def __init__(self, transforms: Optional[Sequence[Dict]] = None):
        """Initialize PoseC3D transformation pipeline composer.
        
        Args:
            transforms: Optional sequence of transformation configuration dictionaries.
                Each dict must contain a 'type' key with the import path, and optional
                additional keys for constructor/function arguments.
                
        Raises:
            ValueError: If a transform dict is missing the 'type' key
            ImportError: If a specified module or attribute cannot be imported
        """
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
        """Execute the transformation pipeline on input data.
        
        Applies each transformation in sequence, passing the output of each
        transform as input to the next. If any transform returns None,
        the pipeline terminates early and returns None.
        
        Args:
            data: Input data dictionary to process through the pipeline
            
        Returns:
            Optional[dict]: Transformed data after applying all pipeline steps,
                or None if any transformation returned None (indicating failure
                or that the data should be filtered out)
        """
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data
