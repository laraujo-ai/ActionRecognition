import logging
from models.recognizers.base import IRecognizer
from models.functions import softmax
from models.preprocessors.posec3d_preprocessor import Posec3dPreprocessor

logger = logging.getLogger(__name__)


class Posec3DRecognizer(IRecognizer):

    def __init__(
        self, preprocessor: Posec3dPreprocessor, model_engine, label_map_path: str
    ):
        try:
            self.model_engine = model_engine
            self.preprocessor = preprocessor
            self.input_name = self.model_engine.get_inputs()[0].name
            self.label_map = [x.strip() for x in open(label_map_path).readlines()]
        except Exception as e:
            logger.error(f"Failed to initialize Posec3DRecognizer: {e}")
            raise

    def inference(self, data):
        try:
            input_ = self.preprocessor.process(data)
            outputs = self.model_engine.run(None, {self.input_name: input_})
            return self.post_process_results(outputs)
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise

    def post_process_results(self, outputs: list):
        batched_results = []
        for output in outputs:
            out = softmax(output, axis=1).mean(axis=0)
            out = out.argmax().item(0)
            predicted_action = self.label_map[out]
            batched_results.append(predicted_action)

        return batched_results[
            0
        ]  # will return only the first video result for now, after if needed we can use batch


class Posec3DFeatureExtractor(IRecognizer):

    def __init__(self, preprocessor, model_engine): ...
    def inference(self, data): ...
