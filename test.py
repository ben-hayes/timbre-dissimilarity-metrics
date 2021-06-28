import torch
from torch.functional import norm

import timbremetrics

if __name__ == "__main__":
    datasets = timbremetrics.list_datasets()
    dataset = datasets[0]
    audio = timbremetrics.get_audio(dataset)

    min_length = 1e9

    embeddings = torch.stack(
        [
            torch.stft(
                torch.tensor(audio_file["audio"]),
                2048,
                window=torch.hann_window(2048),
                return_complex=True,
                normalized=True,
            )
            .abs()
            .mean(dim=-1)
            for audio_file in audio
        ],
        dim=0,
    )

    metric = timbremetrics.TimbreMAE(dataset=dataset, distance=timbremetrics.l2)
    print(metric(embeddings))
