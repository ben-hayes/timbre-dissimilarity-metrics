from math import perm
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


class TimbreMeanErrorMetric(TimbreMetric):
    def __init__(self, dataset=None, distance=l2, dist_sync_on_step=False):
        super().__init__(dataset, distance, dist_sync_on_step)

        error = torch.tensor(0) if self.dataset else []
        self.add_state("error", default=error, dist_reduce_fx="sum")

    def _compute_item_error(self, target: torch.Tensor, distances: torch.Tensor):
        raise NotImplementedError("Can't instantiate abstract base class.")

    def update(self, embeddings: Union[torch.Tensor, dict]):
        self._validate_embeddings(embeddings)
        if not self.dataset:
            for dataset in self.datasets:
                distances = self._compute_embedding_distances(embeddings[dataset])
                target = self.dissimilarity_matrices[dataset]
                error = self._compute_item_error(target, distances)
                self.error.append(error)
        else:
            distances = self._compute_embedding_distances(embeddings)
            target = self.dissimilarity_matrix
            error = self._compute_item_error(target, distances)
            self.error.append(error)

    def compute(self):
        return self.error


class MAE(TimbreMeanErrorMetric):
    def __init__(self, dataset=None, distance=l2, dist_sync_on_step=False):
        super().__init__(dataset, distance, dist_sync_on_step)

    def _compute_item_error(self, target: torch.Tensor, distances: torch.Tensor):
        absolute_error = torch.sum(torch.abs(target - distances))
        count = torch.sum(torch.ones_like(target).triu(1))
        return absolute_error / count


class MSE(TimbreMeanErrorMetric):
    def __init__(self, dataset=None, distance=l2, dist_sync_on_step=False):
        super().__init__(dataset, distance, dist_sync_on_step)

    def _compute_item_error(self, target: torch.Tensor, distances: torch.Tensor):
        squared_error = torch.sum((target - distances) ** 2)
        count = torch.sum(torch.ones_like(target).triu(1))
        return squared_error / count


class TimbreRankedErrorMetric(TimbreMeanErrorMetric):
    def __init__(self, dataset=None, distance=l2, dist_sync_on_step=False):
        super().__init__(dataset, distance, dist_sync_on_step)

    def _get_rankings(self, target: torch.Tensor, distances: torch.Tensor):
        distances_ranked = torch.argsort(distances, dim=0)
        target_ranked = torch.argsort(target, dim=0)

        return target_ranked, distances_ranked


class ItemRankingAgreement(TimbreRankedErrorMetric):
    def __init__(self, k=None, dataset=None, distance=l2, dist_sync_on_step=False):
        super().__init__(dataset, distance, dist_sync_on_step)
        self.k = k

    def _compute_item_error(self, target: torch.Tensor, distances: torch.Tensor):
        target_full = target + target.t()
        distances_full = distances + distances.t()
        target_ranked, distances_ranked = self._get_rankings(
            target_full, distances_full
        )

        if self.k:
            target_ranked = target_ranked[target_ranked <= self.k]
            distances_ranked = distances_ranked[distances_ranked <= self.k]
            comparisons = self.k * target.shape[0]
        else:
            comparisons = distances.numel() - distances.shape[0]

        matching_ranks = (
            torch.sum((target_ranked == distances_ranked).float()) - distances.shape[0]
        )

        return matching_ranks / comparisons


class TripletAgreement(TimbreMeanErrorMetric):
    def __init__(
        self,
        positive_radius=0.3,
        margin=0.1,
        enforce_margin=True,
        dataset=None,
        distance=l2,
        dist_sync_on_step=False,
    ):
        super().__init__(dataset, distance, dist_sync_on_step)
        self.margin = margin
        self.enforce_margin = enforce_margin
        self.positive_radius = positive_radius

    def _compute_item_error(self, target: torch.Tensor, distances: torch.Tensor):
        triplet_agreements = 0
        total_triplets = 0
        for anchor in range(target.shape[0]):
            positives = torch.nonzero(target[anchor] < self.positive_radius)
            negatives = torch.nonzero(
                target[anchor] > self.positive_radius + self.margin
            )
            for positive in positives:
                for negative in negatives:
                    total_triplets += 1
                    margin = self.margin if self.enforce_margin else 0
                    if (
                        distances[anchor, positive] + margin
                        < distances[anchor, negative]
                    ):
                        triplet_agreements += 1

        return (
            torch.tensor(triplet_agreements / total_triplets)
            if total_triplets > 0
            else torch.tensor(float("nan"))
        )


class Mantel(TimbreMeanErrorMetric):
    def __init__(
        self,
        method="pearson",
        permutations=0,
        alternative="greater",
        dataset=None,
        distance=l2,
        dist_sync_on_step=False,
    ):
        super().__init__(dataset, distance, dist_sync_on_step)
        assert method in (
            "pearson",
            "spearman",
        ), 'Method must be one of ("pearson", "spearman")'
        assert alternative in (
            "greater",
            "less",
            "two-sided",
        ), 'Alternative hypothesis must be one of ("greater", "less", "two-sided")'
        self.correlation_function = (
            self._pearsonr if method == "pearson" else self._spearmanr
        )
        self.alternative_hypothesis = (
            (lambda r, permutations: r <= permutations)
            if alternative == "greater"
            else (lambda r, permutations: r >= permutations)
            if alternative == "less"
            else (lambda r, permutations: torch.abs(r) <= torch.abs(permutations))
        )

        self.permutations = permutations

    def _make_upper_tri_mask(self, matrix: torch.Tensor):
        return torch.ones_like(matrix).triu(1).bool()

    def _to_condensed_upper_triangle(self, matrix: torch.Tensor):
        upper_tri_mask = self._make_upper_tri_mask(matrix)
        condensed_matrix = matrix[upper_tri_mask]
        return condensed_matrix

    def _permute_upper_triangle(self, matrix: torch.Tensor):
        upper_tri_mask = self._make_upper_tri_mask(matrix)
        permutation = torch.randperm(matrix[upper_tri_mask].numel())
        matrix = matrix.clone()
        matrix[upper_tri_mask] = matrix[upper_tri_mask][permutation]
        return matrix

    def _to_standardised_condensed_upper_triangle(self, matrix: torch.Tensor):
        condensed_matrix = self._to_condensed_upper_triangle(matrix)

        return (condensed_matrix - condensed_matrix.mean()) / condensed_matrix.std()

    def _to_ranked_condensed_upper_triangle(self, matrix: torch.Tensor):
        condensed_matrix = self._to_condensed_upper_triangle(matrix)

        return condensed_matrix.argsort()

    def _pearsonr(self, a: torch.Tensor, b: torch.Tensor):
        a = self._to_standardised_condensed_upper_triangle(a)
        b = self._to_standardised_condensed_upper_triangle(b)
        numer = torch.sum(a * b)
        denom = a.shape[0]

        return numer / denom

    def _spearmanr(self, a: torch.Tensor, b: torch.Tensor):
        a = self._to_ranked_condensed_upper_triangle(a)
        b = self._to_ranked_condensed_upper_triangle(b)

        denom = a.shape[0] * (a.shape[0] ** 2 - 1)
        numer = 6 * torch.sum(torch.pow(a - b, 2))

        return 1 - (numer / denom)

    def _permutation_test(self, a: torch.Tensor, b: torch.Tensor):
        permutations = [
            self.correlation_function(self._permute_upper_triangle(a), b)
            for _ in range(self.permutations)
        ]
        return torch.tensor(permutations)

    def _compute_item_error(self, target: torch.Tensor, distances: torch.Tensor):
        r = self.correlation_function(target, distances)

        if self.permutations == 0:
            p_value = torch.tensor(float("nan"))
        else:
            permutations = self._permutation_test(distances, target)

            p_value = (
                torch.sum(self.alternative_hypothesis(r, permutations))
                / self.permutations
            )

        return (r, p_value)
