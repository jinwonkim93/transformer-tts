from .kss import KSSLoader
from .feature_extractor import MelAudioFeatureExtractor


def create_dataset(config):
    if config.dataset.type == "kss":
        feature_extractor = MelAudioFeatureExtractor()
        train_dataset = KSSLoader(
            config.dataset.root, split="train", feature_extractor=feature_extractor
        )
        val_dataset = KSSLoader(
            config.dataset.root, split="val", feature_extractor=feature_extractor
        )
    else:
        raise NotImplementedError
    return train_dataset, val_dataset
