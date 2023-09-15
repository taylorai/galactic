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


def remove_cluster(self, cluster: int):
    self.dataset = self.dataset.filter(lambda x: x["__cluster"] != cluster)
    self.cluster_ids.remove(cluster)
    del self.cluster_centers[cluster]


def get_cluster_info(self):
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
        print(nn)
