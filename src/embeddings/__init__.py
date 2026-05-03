from src.embeddings.yamnet import YAMNetExtractor
from src.embeddings.panns import PANNsExtractor
from src.embeddings.birdnet import BirdNetExtractor

def get_extractor(model_name: str):
    if model_name == 'yamnet':
        return YAMNetExtractor()
    elif model_name == 'panns':
        return PANNsExtractor()
    elif model_name == 'birdnet':
        return BirdNetExtractor()
    else:
        raise ValueError(f"Unknown model: {model_name}")
