from typing import Union

import numpy as np
from torchmetrics import Metric
import torch

from .utils import list_datasets, load_dissimilarity_matrix


def l1(a, b):
    return torch.sum(torch.abs(a - b))


def l2(a, b):
    return torch.sqrt(torch.sum((a - b) ** 2))


class TimbreMetric(Metric):
    def __init__(self, dataset=None, distance=l2, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.distance = distance
        self.datasets = list_datasets()
        self.dataset = None
        if not dataset:
            self.dissimilarity_matrices = {}
            for d in self.datasets:
                self.dissimilarity_matrices[d] = torch.tensor(
                    load_dissimilarity_matrix(d)
                )
        elif dataset not in self.datasets:
            raise ValueError(
                "Dataset string does not match one provided in library. "
                + "Get available datasets with timbremetrics.list_datasets()."
            )
        else:
            self.dataset = dataset
            self.dissimilarity_matrix = torch.tensor(load_dissimilarity_matrix(dataset))

    def _validate_embeddings(self, embeddings: Union[torch.Tensor, dict]):
        if self.dataset:
            assert embeddings.shape[0] == self.dissimilarity_matrix.shape[0], (
                "Embeddings must be present for all items in dataset. Get all "
                + "dataset items with timbremetrics.get_audio(<dataset>)"
            )
        else:
            for dataset in self.dissimilarity_matrices:
                assert dataset in embeddings, (
                    "When no dataset is specified, all datasets must be present "
                    + "in embeddings dictionary. Get available datasets with "
                    + "timbremetrics.list_datasets()"
                )
                assert (
                    embeddings[dataset].shape[0]
                    == self.dissimilarity_matrices[dataset].shape[0]
                ), (
                    "Embeddings must be present for all items in all datasets. "
                    + "Get all dataset items with timbremetrics.get_audio()"
                )
            for dataset in embeddings:
                assert dataset in self.dissimilarity_matrices, (
                    "When no dataset is specified, all datasets in embeddings "
                    + "must exist in library. List all available datasets with "
                    + "timbremetrics.list_datasets()"
                )

    def _compute_embedding_distances(self, embeddings: torch.Tensor):
        distances = torch.zeros(embeddings.shape[0], embeddings.shape[0])
        for i in range(embeddings.shape[0] - 1):
            for j in range(i + 1, embeddings.shape[0]):
                distances[i, j] = self.distance(embeddings[i], embeddings[j])
        return distances


class TimbreDistanceErrorMetric(TimbreMetric):
    def __init__(self, dataset=None, distance=l2, dist_sync_on_step=False):
        super().__init__(dataset, distance, dist_sync_on_step)
        self.add_state("error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def _compute_error(self, target: torch.Tensor, distances: torch.Tensor):
        raise NotImplementedError("Can't instantiate abstract base class.")

    def update(self, embeddings: Union[torch.Tensor, dict]):
        self._validate_embeddings(embeddings)
        if not self.dataset:
            for dataset in self.datasets:
                distances = self._compute_embedding_distances(embeddings[dataset])
                dataset_error = self._compute_error(
                    self.dissimilarity_matrices[dataset], distances
                )
                self.error += dataset_error
                self.count += embeddings[dataset].shape[0]
        else:
            distances = self._compute_embedding_distances(embeddings)
            dataset_error = self._compute_error(self.dissimilarity_matrix, distances)
            self.error += dataset_error
            self.count += embeddings.shape[0]

    def compute(self):
        return self.error / self.count


class TimbreMAE(TimbreDistanceErrorMetric):
    def __init__(self, dataset=None, distance=l2, dist_sync_on_step=False):
        super().__init__(dataset, distance, dist_sync_on_step)

    def _compute_error(self, target: torch.Tensor, distances: torch.Tensor):
        absolute_error = torch.sum(torch.abs(target - distances))
        return absolute_error


class TimbreMSE(TimbreDistanceErrorMetric):
    def __init__(self, dataset=None, distance=l2, dist_sync_on_step=False):
        super().__init__(dataset, distance, dist_sync_on_step)

    def _compute_error(self, target: torch.Tensor, distances: torch.Tensor):
        squared_error = torch.sum((target - distances) ** 2)
        return squared_error


class TimbreRankedErrorMetric(TimbreDistanceErrorMetric):
    def __init__(self, dataset=None, distance=l2, dist_sync_on_step=False):
        super().__init__(dataset, distance, dist_sync_on_step)

    def _get_rankings(self, target: torch.Tensor, distances: torch.Tensor):
        distances_full = distances + distances.t()
        target_full = target + target.t()

        distances_ranked = torch.argsort(distances_full, dim=1)
        target_ranked = torch.argsort(target_full, dim=1)

        return target_ranked, distances_ranked


class TimbreRankingDistance(TimbreRankedErrorMetric):
    def __init__(self, dataset=None, distance=l2, dist_sync_on_step=False):
        super().__init__(dataset, distance, dist_sync_on_step)

    def _compute_error(self, target: torch.Tensor, distances: torch.Tensor):
        target_ranked, distances_ranked = self._get_rankings(target, distances)

        sum_absolute_rank_error = torch.sum(
            torch.abs(target_ranked - distances_ranked), dim=1
        )

        return torch.sum(sum_absolute_rank_error)


class TimbreSpearmanCorrCoef(TimbreRankedErrorMetric):
    def __init__(self, dataset=None, distance=l2, dist_sync_on_step=False):
        super().__init__(dataset, distance, dist_sync_on_step)

    def _compute_error(self, target: torch.Tensor, distances: torch.Tensor):
        target_ranked, distances_ranked = self._get_rankings(target, distances)

        squared_rank_difference = (target_ranked - distances_ranked) ** 2
        summed_squared_rank_difference = squared_rank_difference.sum(dim=1)

        n = target.shape[0]

        rho = 1 - (6 * summed_squared_rank_difference) / (n * (n ** 2 - 1))

        return rho.mean()

class TimbreRankAtK(TimbreRankedErrorMetric):

     def __init__(self, dataset=None, distance=l2, dist_sync_on_step=False, k=5):   
         super().__init__(dataset, distance, dist_sync_on_step)

         self.k = k
 
     def _compute_error(self, target: torch.Tensor, distances: torch.Tensor):
         target_ranked, distances_ranked = self._get_rankings(target, distances)
         
         target_ranked_mask = target_ranked <= self.k
         target_ranked = target_ranked * target_ranked_mask.float()

         distances_ranked_mask = distances_ranked <= self.k
         distances_ranked = distances_ranked * distances_ranked_mask.float()           

         rank_difference = torch.sum(torch.abs(target_ranked - distances_ranked), dim=1)
 
         return torch.sum(rank_difference)