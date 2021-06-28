import os
import pkg_resources

from .utils import load_audio, list_datasets


def get_audio(dataset=None):
    datasets = list_datasets()
    if not dataset:
        dataset_audio = {}
        for d in datasets:
            dataset_audio[d] = load_dataset_audio(d)
        return dataset_audio
    elif dataset not in datasets:
        raise ValueError(
            "Dataset string does not match one provided in library. Get available datasets with timbremetrics.list_datasets()."
        )
    else:
        return load_dataset_audio(dataset)


def load_dataset_audio(dataset):
    audio_files = pkg_resources.resource_listdir(
        __name__, os.path.join("sounds", dataset)
    )
    audio_files = sorted(audio_files)

    audio_data = []
    for audio_file in audio_files:
        if os.path.splitext(audio_file)[1] != ".aiff":
            continue

        audio, sr = load_audio(dataset, audio_file)
        audio_data.append({"file": audio_file, "audio": audio, "sample_rate": sr})

    return audio_data
