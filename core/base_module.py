from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any

class BaseVisionModule(ABC):
    """
    Abstract base class for all AI vision modules in the Pi Vision AI plugin system.
    """

    @abstractmethod
    def initialize(self):
        """
        Allocate resources, load weights, or setup variables.
        Called once when the module is activated.
        """
        pass

    @abstractmethod
    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame.
        
        Args:
            frame (np.ndarray): The BGR image frame from the camera.
            
        Returns:
            np.ndarray: The processed frame to be displayed.
        """
        pass

    @abstractmethod
    def cleanup(self):
        """
        Teardown resources, clear memory, close files.
        Called when the module is deactivated or swapped out.
        """
        pass

    def update_settings(self, settings: Dict[str, Any]):
        """
        Update dynamic settings for the module (e.g. confidence threshold).
        Implementations should override this if they expose settings.
        """
        pass

    def handle_interaction(self, data: Dict[str, Any]):
        """
        Process interactions from the UI (e.g. mouse clicks, button presses).
        """
        pass

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """
        Return metadata about the module.
        Must return a dictionary containing at minimum: 'id', 'name', 'description'.
        """
        pass
