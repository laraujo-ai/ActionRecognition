from typing import Protocol, Any
import importlib.util
import os
from types import ModuleType


class BaseParser(Protocol):
    """Protocol interface for configuration file parsers.
    
    Defines the contract that all parsers must implement for loading
    and processing configuration files into usable formats.
    """
    def __init__(self):
        """Initialize the parser."""
        ...
        
    def parse(self, file_path: str) -> Any:
        """Parse a configuration file.
        
        Args:
            file_path: Path to the configuration file to parse
            
        Returns:
            Any: Parsed configuration data
        """
        ...


class Posec3dConfigParser(BaseParser):
    """Parser for PoseC3D Python configuration files.
    
    Dynamically imports Python configuration files as modules, allowing
    access to configuration variables, transformation pipelines, and other
    settings defined in the config files. This enables flexible configuration
    management without requiring JSON or YAML parsing.
    
    The parser can load any valid Python file and return it as a module object,
    making all defined variables and functions accessible as module attributes.
    """
    
    def __init__(self):
        """Initialize PoseC3D configuration parser."""
        ...

    def parse(self, file_path: str) -> ModuleType:
        """Parse a Python configuration file into a module object.
        
        Args:
            file_path: Path to the Python configuration file to load
            
        Returns:
            ModuleType: Loaded configuration as a Python module with all
                variables and functions accessible as attributes
                
        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            ImportError: If the configuration file has syntax errors
        """
        config_module = self.import_module_from_path(file_path)
        return config_module

    def import_module_from_path(self, file_path: str) -> ModuleType:
        """Dynamically import a Python file as a module.
        
        Uses importlib to load and execute a Python file, returning it as
        a module object. This allows access to all variables, functions,
        and classes defined in the file.
        
        Args:
            file_path: Absolute or relative path to the Python file to import
            
        Returns:
            ModuleType: The loaded module with all defined attributes accessible
            
        Raises:
            FileNotFoundError: If the specified file doesn't exist
            ImportError: If the file has syntax errors or import issues
        """
        file_path = os.path.abspath(file_path)
        spec = importlib.util.spec_from_file_location("config_module", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # execute the module
        return module
