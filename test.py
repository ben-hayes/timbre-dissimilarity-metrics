import numpy as np
from sklearn.manifold import MDS
import torch

import timbremetrics
from timbremetrics.utils import load_dissimilarity_matrix

if __name__ == "__main__":
    metrics = [
        ["MSE", timbremetrics.MSE()],
        ["MAE", timbremetrics.MAE()],
        ["Ranking Agreement", timbremetrics.ItemRankingAgreement()],
        ["Ranking Agreement k=5", timbremetrics.ItemRankingAgreement(k=5)],
        ["Triplet Agreement", timbremetrics.TripletAgreement()],
        [
            "Mantel test (pearson)",
            timbremetrics.Mantel(method="pearson", permutations=500),
        ],
        [
            "Mantel test (spearman)",
            timbremetrics.Mantel(method="spearman", permutations=500),
        ],
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
        if isinstance(mds_metric[0], list):
            mds_pvalues = np.array([m[1] for m in mds_metric])
            mds_metric = np.array([m[0] for m in mds_metric])
            rand_pvalues = np.array([m[1] for m in rand_metric])
            rand_metric = np.array([m[0] for m in rand_metric])
            print(
                "MDS : mean %.4f, min %.4f, max %.4f; p: mean %.4f, min %.4f, max %.4f"
                % (
                    mds_metric.mean(),
                    mds_metric.min(),
                    mds_metric.max(),
                    mds_pvalues.mean(),
                    mds_pvalues.min(),
                    mds_pvalues.max(),
                )
            )
            print(
                "Rand: mean %.4f, min %.4f, max %.4f; p: mean %.4f, min %.4f, max %.4f"
                % (
                    rand_metric.mean(),
                    rand_metric.min(),
                    rand_metric.max(),
                    rand_pvalues.mean(),
                    rand_pvalues.min(),
                    rand_pvalues.max(),
                )
            )
        else:
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
