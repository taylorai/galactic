import datasets
import networkx as nx
import numpy as np
from tqdm.auto import tqdm
import random
from typing import Optional

import logging

logger = logging.getLogger("galactic")


def get_duplicates(
    cluster: datasets.Dataset, threshold: float, strategy: str = "random"
):
    duplicates = []
    num_points = len(cluster)
    emb_matrix = np.array(cluster["__embedding"])
    if strategy != "random":
        centroid = np.mean(emb_matrix, axis=0)
        id2dist = {
            cluster[i]["__id"]: np.dot(emb_matrix[i], centroid)
            for i in range(num_points)
        }

    # find connected components
    similarities = np.dot(emb_matrix, emb_matrix.T)
    G = nx.Graph()
    for i in range(num_points):
        for j in range(i + 1, num_points):
            if similarities[i][j] > threshold:
                G.add_edge(
                    cluster[i]["__id"],
                    cluster[j]["__id"],
                )
    # get duplicates
    for cmp in nx.connected_components(G):
        cmp = list(cmp)
        if strategy == "random":
            duplicates.extend(cmp[1:])
        elif strategy == "nearest":
            cmp.sort(key=lambda x: id2dist[x])
            duplicates.extend(cmp[1:])
        elif strategy == "furthest":
            cmp.sort(key=lambda x: -id2dist[x])
            duplicates.extend(cmp[1:])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    return duplicates


def tune_threshold(
    cluster: datasets.Dataset,
    target_retention: float,
    tol: float = 0.01,
    max_iter: int = 30,
):
    tol = max(tol, 1 / len(cluster))
    emb_matrix = np.array(cluster["__embedding"])
    similarities = np.dot(emb_matrix, emb_matrix.T)
    min_sim = np.min(similarities)
    max_sim = np.max(similarities)

    # use binary search to find threshold
    lo = min_sim
    hi = max_sim

    for _ in range(max_iter):
        mid = (lo + hi) / 2
        duplicates = get_duplicates(cluster, mid)
        # Calculate the retention rate after removing duplicates
        retention = 1 - len(duplicates) / len(cluster)
        print(f"Threshold: {round(mid, 3)}, Retention: {round(retention, 3)}")

        # if retention is within tolerance, we're done
        if abs(retention - target_retention) < tol:
            return mid
        # if retention is too low, increase the threshold (less filtering)
        elif retention < target_retention:
            lo = mid
        # if retention is too high, lower the threshold (more filtering)
        elif retention > target_retention:
            hi = mid

    # Final threshold is the average of the final bounds
    final_threshold = (hi + lo) / 2
    return final_threshold


def semdedup(
    self,
    target_retention: Optional[float] = 0.8,
    threshold: Optional[float] = None,
    inplace=True,
):
    if target_retention is None and threshold is None:
        raise ValueError(
            "You must specify either target_retention or threshold."
        )
    if target_retention is not None and threshold is not None:
        logger.warning(
            "Both target_retention and threshold specified. Using target_retention to tune threshold."
        )

    if target_retention is not None:
        cluster_ids = list(set(self.dataset["__cluster"]))
        tuning_clusters = random.choices(cluster_ids, k=3)
        # tune threshold
        logger.info("Tuning threshold on 3 clusters...")
        thresholds = []
        for tuning_cluster in tuning_clusters:
            threshold = tune_threshold(
                self.dataset.filter(
                    lambda x: x["__cluster"] == tuning_cluster
                ),
                target_retention,
            )
            thresholds.append(threshold)
        threshold = np.mean(thresholds)
        logger.info(f"Threshold: {round(threshold, 2)}")

    # get duplicates
    remove = []
    for cluster_id in self.cluster_ids:
        cluster = self.dataset.filter(lambda x: x["__cluster"] == cluster_id)
        if len(cluster) < 2:
            continue
        duplicates = get_duplicates(cluster, threshold)
        logger.info(
            f"Cluster {cluster_id} has {len(duplicates)} duplicates ({round(len(duplicates) / len(cluster) * 100, 1)}%).\n"
        )
        remove.extend(duplicates)

    if inplace:
        before_dedup = len(self.dataset)
        self.dataset = self.dataset.filter(lambda x: x["__id"] not in remove)
        n_removed = len(remove)
        logger.info(
            f"Removed {len(remove)} / {before_dedup} items flagged as semantic near-duplicates ({round(n_removed / before_dedup * 100, 2)}%)."
        )
        return self
    else:
        before_dedup = len(self.dataset)
        new_dataset = self.dataset.filter(lambda x: x["__id"] not in remove)
        n_removed = len(remove)
        logger.info(
            f"Removed {len(remove)} / {before_dedup} items flagged as semantic near-duplicates ({round(n_removed / before_dedup * 100, 2)}%)."
        )
        return type(self)(
            new_dataset,
            model=self.model,
            emb_matrix=self.emb_matrix,
            cluster_ids=self.cluster_ids,
            cluster_centers=self.cluster_centers,
            openai_api_key=self.openai_api_key,
        )
