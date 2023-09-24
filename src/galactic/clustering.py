import datasets
import networkx as nx
import numpy as np
import random
from typing import Optional
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
from collections import Counter

import logging

logger = logging.getLogger("galactic")


def cluster(
    self,
    n_clusters: int,
    method: str = "kmeans",
    batch_size: int = 1024,
    n_epochs: int = 5,
):
    if "__embedding" not in self.dataset.column_names:
        raise ValueError(
            "You must call get_embeddings() before calling cluster(). If your dataset already has an embeddings column, make sure it's named '__embeddings'."
        )
    if method == "minibatch_kmeans":
        model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size)
        for epoch in range(n_epochs):
            logger.info(f"Epoch {epoch+1}/{n_epochs}")
            self.dataset.map(
                lambda x: model.partial_fit(np.array(x["__embedding"])),
            )
        self.cluster_ids = list(range(n_clusters))
        # cluster centers is a dict of id -> center
        self.cluster_centers = {
            i: model.cluster_centers_[i] for i in range(n_clusters)
        }

    elif method == "kmeans":
        model = KMeans(n_clusters=n_clusters, init="k-means++", n_init=1)
        arr = np.array(self.dataset["__embedding"])
        model.fit(arr)
        self.cluster_ids = list(range(n_clusters))
        # cluster centers is a dict of id -> center
        self.cluster_centers = {
            i: model.cluster_centers_[i] for i in range(n_clusters)
        }
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    # add new column with cluster labels
    self.dataset = self.dataset.map(
        lambda x: {"__cluster": model.predict(x["__embedding"])},
        batched=True,
    )


# preferred to filtering out the cluster, because it will remove the cluster from the cluster_ids list
def remove_cluster(self, cluster: int):
    self.dataset = self.dataset.filter(lambda x: x["__cluster"] != cluster)
    self.cluster_ids.remove(cluster)
    del self.cluster_centers[cluster]


def get_cluster_info(self, n_neighbors: int = 3, field: str = None):
    """
    Goal is to do some kind of unsupervised domain discovery thing here to figure out what the clusters mean.
    """
    if not hasattr(self, "cluster_centers"):
        raise ValueError(
            "You must call cluster() before calling get_cluster_info()"
        )
    if not hasattr(self, "model"):
        raise ValueError(
            "You must call get_embeddings() before calling get_cluster_info()"
        )

    counts = Counter(self.dataset["__cluster"])

    # for each one, get the 3 nearest neighbors
    for id, emb in self.cluster_centers.items():
        print(f"Cluster {id} ({counts[id]} items)")
        nn = self.get_nearest_neighbors(emb, k=3)
        if field is not None:
            for n in nn:
                print("\t" + n[field])
                print("---")
        else:
            for n in nn:
                print({k: v for k, v in n.items() if k != "__embedding"})


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
