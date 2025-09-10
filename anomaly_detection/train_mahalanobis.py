import sys
import json
from pathlib import Path
from typing import Dict, Any
import yaml
import argparse

from anomaly_detection.data.data_loader import VideoDataLoader, DatasetConfig
from anomaly_detection.trainers.mahalanobis_trainer import MahalanobisTrainer
from anomaly_detection.data.schemas import TrainingConfig


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and parse YAML configuration file.

    Args:
        config_path: Path to YAML configuration file containing training parameters

    Returns:
        Dict[str, Any]: Parsed configuration dictionary with all training settings

    Raises:
        RuntimeError: If configuration file cannot be loaded or parsed
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise RuntimeError(f"Failed to load config from {config_path}: {e}")


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration structure and required parameters.

    Ensures all required configuration sections and keys are present
    for successful training execution.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If required configuration sections or keys are missing
    """
    required_sections = ["dataset", "model", "training"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    dataset_config = config["dataset"]
    required_dataset_keys = ["base_path", "normal_categories", "anomaly_categories"]
    for key in required_dataset_keys:
        if key not in dataset_config:
            raise ValueError(f"Missing required dataset config key: {key}")

    model_config = config["model"]
    if "config_path" not in model_config:
        raise ValueError("Missing required model config key: config_path")


def create_dataset_config(config: Dict[str, Any]) -> DatasetConfig:
    """Create structured dataset configuration from raw config dictionary.

    Args:
        config: Raw configuration dictionary containing dataset parameters

    Returns:
        DatasetConfig: Structured dataset configuration object
    """
    dataset_config = config["dataset"]

    return DatasetConfig(
        base_path=dataset_config["base_path"],
        normal_categories=dataset_config["normal_categories"],
        anomaly_categories=dataset_config["anomaly_categories"],
        limits=dataset_config.get("limits", {}),
    )


def create_training_config(config: Dict[str, Any]) -> TrainingConfig:
    """Create structured training configuration from raw config dictionary.

    Extracts training parameters and PCA settings with sensible defaults
    for Mahalanobis distance-based anomaly detection.

    Args:
        config: Raw configuration dictionary containing training parameters

    Returns:
        TrainingConfig: Structured training configuration object
    """
    training_config = config["training"]
    pca_config = config["model"].get("pca", {})

    return TrainingConfig(
        validation_percentile=training_config.get("validation_percentile", 95),
        covariance_estimator=training_config.get("covariance_estimator", "ledoit_wolf"),
        fallback_threshold_std_multiplier=training_config.get(
            "fallback_threshold_std_multiplier", 3.0
        ),
        enable_pca=pca_config.get("enable", True),
        max_components=pca_config.get("max_components", 64),
        min_samples_ratio=pca_config.get("min_samples_ratio", 0.8),
    )


def save_results(results: Dict[str, Any], filepath: str) -> None:
    """Save training results and metrics to JSON file.

    Creates parent directories if needed and serializes results with
    proper formatting for analysis and reporting.

    Args:
        results: Complete training results including metrics and model info
        filepath: Output path for JSON results file
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)


def main():
    """Execute complete anomaly detection training workflow.

    Orchestrates the full training pipeline including configuration loading,
    data processing, model training, evaluation, and result persistence.
    Provides comprehensive logging and error handling throughout the process.
    """
    parser = argparse.ArgumentParser(
        description="Train Mahalanobis anomaly detection model"
    )
    parser.add_argument("config_path", help="Path to training configuration YAML file")
    args = parser.parse_args()

    print("Mahalanobis Anomaly Detection Training")
    print("-" * 40)

    try:
        print(f"Loading configuration: {args.config_path}")
        config = load_config(args.config_path)
        validate_config(config)
        print("Configuration validated")

        dataset_config = create_dataset_config(config)
        training_config = create_training_config(config)
        model_config = config["model"]
        output_config = config.get("output", {})
        verbose = output_config.get("verbose", True)

        print("\nInitializing data loader")
        data_loader = VideoDataLoader(
            config=dataset_config, model_config=model_config, verbose=verbose
        )

        print("Loading dataset")
        X_train, (X_val, y_val), (X_test, y_test) = data_loader.load_dataset()

        print(f"\nDataset loaded:")
        print(f"  Training: {X_train.shape[0]} samples")
        print(f"  Validation: {X_val.shape[0]} samples")
        print(f"  Test: {X_test.shape[0]} samples")
        print(f"  Features: {X_train.shape[1] if X_train.size > 0 else 'N/A'}")

        print("\nInitializing trainer")
        trainer = MahalanobisTrainer(config=training_config, verbose=verbose)

        print("\nTraining model...")
        trained_model = trainer.train(X_train, X_val, y_val, X_test)

        print("\nEvaluating model...")
        evaluation_results = trainer.evaluate(trained_model, X_test, y_test)

        if output_config.get("save_model", False):
            model_path = output_config.get(
                "model_save_path", "./trained_models/mahalanobis_model.pkl"
            )
            print(f"\nSaving model: {model_path}")
            trained_model.save(model_path)
            print("Model saved")

        if output_config.get("save_results", False):
            results_path = output_config.get(
                "results_save_path", "./results/training_results.json"
            )
            print(f"\nSaving results: {results_path}")

            full_results = {
                "config": config,
                "dataset_summary": {
                    "train_samples": X_train.shape[0] if X_train.size > 0 else 0,
                    "val_samples": X_val.shape[0] if X_val.size > 0 else 0,
                    "test_samples": X_test.shape[0] if X_test.size > 0 else 0,
                    "feature_dim": X_train.shape[1] if X_train.size > 0 else 0,
                    "val_normal": (y_val == 0).sum() if y_val.size > 0 else 0,
                    "val_anomalous": (y_val == 1).sum() if y_val.size > 0 else 0,
                    "test_normal": (y_test == 0).sum() if y_test.size > 0 else 0,
                    "test_anomalous": (y_test == 1).sum() if y_test.size > 0 else 0,
                },
                "model_info": trained_model.get_model_info(),
                "evaluation": evaluation_results,
            }

            save_results(full_results, results_path)
            print("Results saved")

        print("\nTraining completed successfully")

        print(f"\nModel summary:")
        model_info = trained_model.get_model_info()
        for key, value in model_info.items():
            print(f"  {key}: {value}")

        if evaluation_results and "metrics" in evaluation_results:
            print(f"\nTest performance:")
            metrics = evaluation_results["metrics"]
            for metric, value in metrics.items():
                if metric != "confusion_matrix" and value is not None:
                    print(f"  {metric}: {value:.4f}")

    except Exception as e:
        print(f"\nTraining failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
