from anomaly_detection.data.schemas import TrainedModel


def get_trained_mahalanobis(pretrained_path: str) -> TrainedModel:
    model = TrainedModel.load(pretrained_path)
    return model
