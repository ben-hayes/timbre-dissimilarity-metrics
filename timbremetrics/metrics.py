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

    def _compute_error(self, target: torch.Tensor, embeddings: torch.Tensor):
        raise NotImplementedError("Can't instantiate abstract base class.")

    def update(self, embeddings: Union[torch.Tensor, dict]):
        self._validate_embeddings(embeddings)
        if not self.dataset:
            for dataset in self.datasets:
                dataset_error = self._compute_error(
                    self.dissimilarity_matrices[dataset], embeddings[dataset]
                )
                self.error += dataset_error
                self.count += embeddings[dataset].shape[0]
        else:
            dataset_error = self._compute_error(self.dissimilarity_matrix, embeddings)
            self.error += dataset_error
            self.count += embeddings.shape[0]

    def compute(self):
        return self.error / self.count


class TimbreMAE(TimbreDistanceErrorMetric):
    def __init__(self, dataset=None, distance=l2, dist_sync_on_step=False):
        super().__init__(dataset, distance, dist_sync_on_step)

    def _compute_error(self, target: torch.Tensor, embeddings: torch.Tensor):
        distances = self._compute_embedding_distances(embeddings)
        absolute_error = torch.sum(torch.abs(target - distances))
        return absolute_error


class TimbreMSE(TimbreDistanceErrorMetric):
    def __init__(self, dataset=None, distance=l2, dist_sync_on_step=False):
        super().__init__(dataset, distance, dist_sync_on_step)

    def _compute_error(self, target: torch.Tensor, embeddings: torch.Tensor):
        distances = self._compute_embedding_distances(embeddings)
        squared_error = torch.sum((target - distances) ** 2)
        return squared_error


class TimbreRankingDistance(TimbreDistanceErrorMetric):
    def __init__(self, dataset=None, distance=l2, dist_sync_on_step=False):
        super().__init__(dataset, distance, dist_sync_on_step)

    def _compute_error(self, target: torch.Tensor, embeddings: torch.Tensor):
        distances = self._compute_embedding_distances(embeddings)

        distances = distances + distances.t()
        target = target + target.t()

        distances = torch.argsort(distances, dim=1)
        target = torch.argsort(target, dim=1)

        item_rank_scores = torch.sum(torch.abs(target - distances), dim=1)

        return torch.sum(item_rank_scores)