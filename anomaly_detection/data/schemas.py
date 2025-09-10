from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class DatasetConfig:
    """Dataset configuration for anomaly detection training.

    Defines dataset structure, category mappings, and data limits for training
    pose-based anomaly detection models. Supports configurable sample limits
    for rapid prototyping and development.

    Attributes:
        base_path: Root directory containing train/validation/test subdirectories
        normal_categories: List of action categories considered normal behavior
        anomaly_categories: List of action categories considered anomalous behavior
        limits: Dictionary mapping split names to maximum sample counts
    """

    base_path: str
    normal_categories: List[str]
    anomaly_categories: List[str]
    limits: Dict[str, int]


@dataclass
class TrainingConfig:
    """Training configuration for Mahalanobis anomaly detection.

    Controls training hyperparameters, preprocessing options, and algorithm
    settings for Mahalanobis distance-based anomaly detection training.

    Attributes:
        validation_percentile: Percentile for threshold setting on normal validation data
        covariance_estimator: Type of covariance estimator ('ledoit_wolf' or 'empirical')
        fallback_threshold_std_multiplier: Multiplier for fallback threshold calculation
        enable_pca: Whether to apply PCA dimensionality reduction
        max_components: Maximum number of PCA components to retain
        min_samples_ratio: Minimum ratio of samples to features before applying PCA
    """

    validation_percentile: int = 95
    covariance_estimator: str = "ledoit_wolf"
    fallback_threshold_std_multiplier: float = 3.0
    enable_pca: bool = True
    max_components: int = 64
    min_samples_ratio: float = 0.8


@dataclass
class TrainedModel:
    """Trained Mahalanobis anomaly detection model container.

    Encapsulates all components of a trained Mahalanobis distance-based anomaly
    detection model, including preprocessing transformers, statistical parameters,
    and detection threshold. Provides model persistence and metadata access.

    Attributes:
        scaler: Fitted StandardScaler for feature normalization
        pca: Fitted PCA transformer for dimensionality reduction (None if not used)
        mean: Mean vector of normal behavior distribution
        covariance: Covariance matrix of normal behavior distribution
        inv_covariance: Pseudo-inverse of covariance matrix for distance computation
        threshold: Anomaly detection threshold for classification
        feature_dim: Final feature dimensionality after preprocessing
        pca_components: Number of PCA components used (None if PCA not applied)
    """

    scaler: StandardScaler
    pca: Optional[PCA]
    mean: np.ndarray
    covariance: np.ndarray
    inv_covariance: np.ndarray
    threshold: float
    feature_dim: int
    pca_components: Optional[int]

    def save(self, filepath: str) -> None:
        """Save trained model using JSON + NumPy format.

        Creates parent directories if they don't exist and saves the model
        components as separate files for better portability and debugging.

        Args:
            filepath: Base path where to save the model files (without extension)
        """
        base_path = Path(filepath).with_suffix("")
        base_path.parent.mkdir(parents=True, exist_ok=True)

        # Save sklearn objects as separate pickle files
        import pickle

        scaler_path = f"{base_path}_scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

        pca_path = None
        if self.pca is not None:
            pca_path = f"{base_path}_pca.pkl"
            with open(pca_path, "wb") as f:
                pickle.dump(self.pca, f)

        # Save numpy arrays
        np.save(f"{base_path}_mean.npy", self.mean)
        np.save(f"{base_path}_covariance.npy", self.covariance)
        np.save(f"{base_path}_inv_covariance.npy", self.inv_covariance)

        # Save metadata as JSON
        metadata = {
            "threshold": float(self.threshold),
            "feature_dim": int(self.feature_dim),
            "pca_components": int(self.pca_components) if self.pca_components else None,
            "has_pca": self.pca is not None,
            "scaler_path": scaler_path,
            "pca_path": pca_path,
            "mean_path": f"{base_path}_mean.npy",
            "covariance_path": f"{base_path}_covariance.npy",
            "inv_covariance_path": f"{base_path}_inv_covariance.npy",
        }

        with open(f"{base_path}.json", "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "TrainedModel":
        """Load trained model from JSON + NumPy format.

        Args:
            filepath: Base path to the saved model files (without extension)

        Returns:
            TrainedModel: Loaded model ready for inference
        """
        base_path = Path(filepath).with_suffix("")

        # Load metadata
        with open(f"{base_path}.json", "r") as f:
            metadata = json.load(f)

        # Load sklearn objects
        import pickle

        with open(metadata["scaler_path"], "rb") as f:
            scaler = pickle.load(f)

        pca = None
        if metadata["has_pca"]:
            with open(metadata["pca_path"], "rb") as f:
                pca = pickle.load(f)

        # Load numpy arrays
        mean = np.load(metadata["mean_path"])
        covariance = np.load(metadata["covariance_path"])
        inv_covariance = np.load(metadata["inv_covariance_path"])

        return cls(
            scaler=scaler,
            pca=pca,
            mean=mean,
            covariance=covariance,
            inv_covariance=inv_covariance,
            threshold=metadata["threshold"],
            feature_dim=metadata["feature_dim"],
            pca_components=metadata["pca_components"],
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata and configuration information.

        Returns:
            Dict[str, Any]: Model information including dimensions, components, and parameters
        """
        return {
            "feature_dim": self.feature_dim,
            "pca_enabled": self.pca is not None,
            "pca_components": self.pca_components,
            "threshold": self.threshold,
            "mean_vector_shape": self.mean.shape,
            "covariance_matrix_shape": self.covariance.shape,
        }
