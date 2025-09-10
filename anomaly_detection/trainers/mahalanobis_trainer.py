from typing import Dict, Any, Optional, Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf, EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn import metrics

from anomaly_detection.data.schemas import TrainingConfig, TrainedModel
from anomaly_detection.utils.functions import compute_mahalanobis_scores


class MahalanobisTrainer:
    """Mahalanobis distance anomaly detection trainer.

    This class implements training for Mahalanobis distance-based anomaly detection
    using pose-based features. It models normal behavior distribution through
    multivariate Gaussian estimation with shrinkage covariance and applies
    statistical distance thresholding for anomaly detection.

    The training workflow:
    1. Standardize features using Z-score normalization
    2. Apply PCA dimensionality reduction if needed for numerical stability
    3. Estimate multivariate Gaussian distribution of normal behavior
    4. Determine anomaly threshold using validation data statistics
    5. Evaluate performance on test data with comprehensive metrics

    Attributes:
        config: Training configuration including PCA settings and thresholds
        verbose: Enable detailed logging and progress tracking during training
    """

    def __init__(self, config: TrainingConfig, verbose: bool = True):
        """Initialize Mahalanobis trainer with configuration.

        Args:
            config: Training configuration including PCA and covariance settings
            verbose: Enable detailed logging and progress tracking
        """
        self.config = config
        self.verbose = verbose

    def _print(self, message: str) -> None:
        """Print message if verbose mode is enabled.

        Args:
            message: Text message to print if verbose logging is active
        """
        if self.verbose:
            print(message)

    def _get_covariance_estimator(self):
        """Create covariance estimator based on configuration.

        Returns:
            Configured sklearn covariance estimator (LedoitWolf or EmpiricalCovariance)

        Raises:
            ValueError: If unknown covariance estimator type is specified
        """
        if self.config.covariance_estimator == "ledoit_wolf":
            return LedoitWolf()
        elif self.config.covariance_estimator == "empirical":
            return EmpiricalCovariance()
        else:
            raise ValueError(
                f"Unknown covariance estimator: {self.config.covariance_estimator}"
            )

    def _apply_pca(
        self, x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[PCA]]:
        """Apply PCA dimensionality reduction for numerical stability.

        Reduces feature dimensionality when the number of features is large
        relative to training samples, improving covariance matrix conditioning
        and computational efficiency for Mahalanobis distance calculation.

        Args:
            x_train: Training feature matrix of shape (n_samples, n_features)
            x_val: Validation feature matrix of shape (n_val, n_features)
            x_test: Test feature matrix of shape (n_test, n_features)

        Returns:
            Tuple containing:
                - Transformed training features
                - Transformed validation features
                - Transformed test features
                - Fitted PCA transformer (None if PCA not applied)
        """
        n_samples, n_features = x_train.shape

        if not self.config.enable_pca:
            return x_train, x_val, x_test, None

        if n_features <= max(
            self.config.min_samples_ratio * n_samples, self.config.max_components
        ):
            self._print(f"PCA not needed: {n_features} features, {n_samples} samples")
            return x_train, x_val, x_test, None

        n_components = int(min(max(2, n_samples - 1), self.config.max_components))
        self._print(f"Applying PCA: {n_features} -> {n_components} components")

        pca = PCA(n_components=n_components, random_state=42)
        x_train_pca = pca.fit_transform(x_train)
        x_val_pca = (
            pca.transform(x_val) if x_val.size > 0 else np.zeros((0, n_components))
        )
        x_test_pca = (
            pca.transform(x_test) if x_test.size > 0 else np.zeros((0, n_components))
        )

        explained_variance = pca.explained_variance_ratio_.sum()
        self._print(f"PCA explained variance ratio: {explained_variance:.4f}")

        return x_train_pca, x_val_pca, x_test_pca, pca

    def _determine_threshold(
        self,
        x_val: np.ndarray,
        y_val: np.ndarray,
        x_train: np.ndarray,
        mean: np.ndarray,
        inv_cov: np.ndarray,
    ) -> float:
        """Determine the anomaly threshold using validation data or fallback."""

        val_scores = compute_mahalanobis_scores(x_val, mean, inv_cov)
        if val_scores.size > 0 and y_val.size > 0:
            # Use only normal validation samples for threshold
            normal_mask = y_val == 0
            if normal_mask.sum() > 0:
                threshold = np.percentile(
                    val_scores[normal_mask], self.config.validation_percentile
                )
                self._print(
                    f"Threshold from validation normals (p{self.config.validation_percentile}): {threshold:.4f}"
                )
                return threshold
            else:
                self._print(
                    "Warning: No normal samples in validation. Using all validation scores."
                )
                threshold = np.percentile(val_scores, self.config.validation_percentile)
                return threshold

        # Fallback: use training data statistics -> only necessary when no validation data available(not very good)
        train_scores = compute_mahalanobis_scores(x_train, mean, inv_cov)
        threshold = np.mean(
            train_scores
        ) + self.config.fallback_threshold_std_multiplier * np.std(train_scores)
        self._print(f"Fallback threshold from training data: {threshold:.4f}")

        return threshold

    def train(
        self,
        x_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        x_test: np.ndarray,
    ) -> TrainedModel:
        """Train the Mahalanobis anomaly detection model."""

        if x_train.size == 0:
            raise ValueError("No training data provided")

        n_samples, n_features = x_train.shape
        self._print(f"Training data shape: ({n_samples}, {n_features})")

        # Step 1: Standardize features
        self._print("Step 1: Standardizing features...")
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = (
            scaler.transform(x_val) if x_val.size > 0 else np.zeros((0, n_features))
        )
        x_test_scaled = (
            scaler.transform(x_test) if x_test.size > 0 else np.zeros((0, n_features))
        )

        # Step 2: Apply PCA if needed
        self._print("Step 2: Applying PCA (if needed)...")
        x_train_proc, x_val_proc, _, pca = self._apply_pca(
            x_train_scaled, x_val_scaled, x_test_scaled
        )

        final_dim = x_train_proc.shape[1]
        self._print(f"Final feature dimension: {final_dim}")

        # Step 3: Fit covariance estimator
        self._print(
            f"Step 3: Fitting {self.config.covariance_estimator} covariance estimator..."
        )
        cov_estimator = self._get_covariance_estimator()
        cov_estimator.fit(x_train_proc)

        mean = cov_estimator.location_
        covariance = cov_estimator.covariance_

        # Use pseudo-inverse for numerical stability
        inv_covariance = np.linalg.pinv(covariance)

        self._print(f"Mean vector shape: {mean.shape}")
        self._print(f"Covariance matrix shape: {covariance.shape}")
        self._print(
            f"Covariance matrix condition number: {np.linalg.cond(covariance):.2e}"
        )

        # Step 4: Determine threshold
        self._print("Step 4: Determining anomaly threshold...")
        threshold = self._determine_threshold(
            x_val_proc, y_val, x_train_proc, mean, inv_covariance
        )

        trained_model = TrainedModel(
            scaler=scaler,
            pca=pca,
            mean=mean,
            covariance=covariance,
            inv_covariance=inv_covariance,
            threshold=threshold,
            feature_dim=final_dim,
            pca_components=pca.n_components_ if pca is not None else None,
        )

        self._print("Training completed successfully!")
        self._print(f"Model info: {trained_model.get_model_info()}")

        return trained_model

    def evaluate(
        self, model: TrainedModel, x_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate the trained model on test data."""
        if x_test.size == 0 or y_test.size == 0:
            self._print("Warning: No test data for evaluation")
            return {}

        self._print("Evaluating model on test data...")
        x_test_scaled = model.scaler.transform(x_test)
        x_test_proc = (
            model.pca.transform(x_test_scaled)
            if model.pca is not None
            else x_test_scaled
        )

        scores = compute_mahalanobis_scores(
            x_test_proc, model.mean, model.inv_covariance
        )
        predictions = (scores > model.threshold).astype(int)

        results = {
            "threshold": model.threshold,
            "scores": scores.tolist(),
            "predictions": predictions.tolist(),
            "ground_truth": y_test.tolist(),
            "metrics": {},
        }

        results["metrics"]["accuracy"] = metrics.accuracy_score(y_test, predictions)
        results["metrics"]["precision"] = metrics.precision_score(
            y_test, predictions, zero_division=0
        )
        results["metrics"]["recall"] = metrics.recall_score(
            y_test, predictions, zero_division=0
        )
        results["metrics"]["f1"] = metrics.f1_score(
            y_test, predictions, zero_division=0
        )

        try:
            results["metrics"]["auroc"] = metrics.roc_auc_score(y_test, scores)
        except ValueError:
            results["metrics"]["auroc"] = None

        try:
            results["metrics"]["average_precision"] = metrics.average_precision_score(
                y_test, scores
            )
        except ValueError:
            results["metrics"]["average_precision"] = None

        cm = metrics.confusion_matrix(y_test, predictions)
        results["metrics"]["confusion_matrix"] = cm.tolist()

        self._print("Evaluation Results:")
        for metric, value in results["metrics"].items():
            if metric != "confusion_matrix" and value is not None:
                self._print(f"  {metric}: {value:.4f}")

        self._print(f"  Confusion Matrix: {cm.tolist()}")

        return results
