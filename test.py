import numpy as np
from sklearn.manifold import MDS
import torch

import timbremetrics
from timbremetrics.utils import load_dissimilarity_matrix

if __name__ == "__main__":
    metrics = [
        ["MSE", timbremetrics.MSE()],
        ["MAE", timbremetrics.MAE()],
        ["Ranking Agreement", timbremetrics.RankingAgreement()],
        ["Ranking Agreement k=5", timbremetrics.RankingAgreement(k=3)],
        ["Triplet Agreement", timbremetrics.TripletAgreement()],
        ["Mantel test (pearson)", timbremetrics.Mantel(method="pearson")],
        ["Mantel test (spearman)", timbremetrics.Mantel(method="spearman")],
    ]
    datasets = timbremetrics.list_datasets()
    audio = timbremetrics.get_audio()

    min_length = 1e9

    embeddings_mds = {}
    embeddings_random = {}
    for dataset in datasets:
        mds = MDS(n_components=3, dissimilarity="precomputed")
        dissimilarity_matrix = load_dissimilarity_matrix(dataset)
        dissimilarity_matrix += dissimilarity_matrix.T

        embedding = mds.fit_transform(dissimilarity_matrix)
        embedding = torch.tensor(embedding)

        embeddings_mds[dataset] = embedding
        embeddings_random[dataset] = torch.rand_like(embedding)

    for name, metric in metrics:
        print(name)
        mds_metric = metric(embeddings_mds)
        rand_metric = metric(embeddings_random)
        if hasattr(mds_metric[0], "isnan"):
            mds_metric = [m for m in mds_metric if not m.isnan()]
            rand_metric = [m for m in rand_metric if not m.isnan()]
        print(
            "MDS : mean %.4f, min %.4f, max %.8f"
            % (
                np.array(mds_metric).mean(),
                np.array(mds_metric).min(),
                np.array(mds_metric).max(),
            )
        )
        print(
            "Rand: mean %.4f, min %.4f, max %.4f"
            % (
                np.array(rand_metric).mean(),
                np.array(rand_metric).min(),
                np.array(rand_metric).max(),
            )
        )
