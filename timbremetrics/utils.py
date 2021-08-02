import aifc
import os
import pkg_resources

import numpy as np

from .config import AUDIO_DIR, DATA_DIR


def list_datasets():
    dataset_files = [
        f.replace("_dissimilarity_matrix.txt", "")
        for f in pkg_resources.resource_listdir(__name__, DATA_DIR)
    ]
    return sorted(dataset_files)


def load_audio(dataset, audio_file):
    f = pkg_resources.resource_stream(
        __name__, os.path.join(AUDIO_DIR, dataset, audio_file)
    )
    aif = aifc.open(f)
    sample_size = {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}[
        aif.getsampwidth()
    ]
    print(sample_size)

    sr = aif.getframerate()
    n_frames = aif.getnframes()
    audio_bytes = aif.readframes(n_frames)
    print(np.fromstring(audio_bytes, sample_size).max())
    audio = np.fromstring(audio_bytes, sample_size)

    return audio, sr


def load_dissimilarity_matrix(dataset):
    f = pkg_resources.resource_stream(
        __name__, os.path.join(DATA_DIR, "%s_dissimilarity_matrix.txt" % dataset)
    )
    return np.loadtxt(f)
