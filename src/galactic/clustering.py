import datasets
import networkx as nx
import numpy as np
import random
from typing import Optional, Literal
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans, HDBSCAN, BisectingKMeans
from collections import Counter
import jinja2
from .async_openai import run_chat_queries_with_openai

import logging

logger = logging.getLogger("galactic")


def cluster(
    self,
    n_clusters: int, 
    method: Literal[
        "kmeans", "minibatch_kmeans", "bisecting_kmeans", "hdbscan"
    ] = "kmeans",
    embedding_field: str = "__embedding",
    **kwargs,
):
    """
    Cluster the dataset using the specified method.

    .. code-block:: python

        # Cluster your dataset into 10 clusters using minibatch k-means
        ds.cluster(n_clusters=10)

    :param n_clusters: Required. The number of clusters to form.
    :param method: Optional. The clustering method to use. Default = ``kmeans``.
    :param embedding_field: Optional. Specify a name for the field to use for clustering. Default = ``__embedding``.
    :param kwargs: Optional. Additional keyword arguments to pass to the clustering algorithm.

    """


    if embedding_field not in self.dataset.column_names:
        raise ValueError(
            "You must call get_embeddings() before calling cluster(). If your dataset already has an embeddings column, pass it as 'embedding_field' argument."
        )
    # check if emb dimension is large
    if len(self.dataset[embedding_field][0]) >= 384:
        logger.info(
            "Embedding dimension is large, which is fine! But consider also experimenting with dimensionality reduction before clustering."
        )
    if method == "minibatch_kmeans":
        model = MiniBatchKMeans(
            n_clusters=n_clusters, batch_size=kwargs.get("batch_size", 1024)
        )
        for epoch in range(kwargs.get("n_epochs", 5)):
            logger.info(f"Epoch {epoch+1}/{kwargs.get('n_epochs', 5)}")
            self.dataset.map(
                lambda x: model.partial_fit(np.array(x[embedding_field])),
            )
        self.cluster_ids = list(range(n_clusters))
        # cluster centers is a dict of id -> center
        self.cluster_centers = {
            i: model.cluster_centers_[i] for i in range(n_clusters)
        }

    elif method == "kmeans":
        model = KMeans(n_clusters=n_clusters, init="k-means++", n_init=1)
        arr = np.array(self.dataset[embedding_field])
        model.fit(arr)
        self.cluster_ids = list(range(n_clusters))
        # cluster centers is a dict of id -> center
        self.cluster_centers = {
            i: model.cluster_centers_[i] for i in range(n_clusters)
        }
    elif method == "bisecting_kmeans":
        model = BisectingKMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=1,
            bisecting_strategy="largest_cluster",
        )
        arr = np.array(self.dataset[embedding_field])
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
        lambda x: {"__cluster": model.predict(x[embedding_field])},
        batched=True,
    )


# preferred to filtering out the cluster, because it will remove the cluster from the cluster_ids list
def remove_cluster(self, cluster: int):
    """
    Remove a cluster from the dataset.

    .. code-block:: python

        # Remove cluster 2 from the dataset
        ds.remove_cluster(cluster:2)

    :param cluster: The cluster ID to be removed from the dataset.
    """
    self.dataset = self.dataset.filter(lambda x: x["__cluster"] != cluster)
    self.cluster_ids.remove(cluster)
    del self.cluster_centers[cluster]


def ai_label_clusters(
    self,
    fields: list[str],
    new_column: str = "__cluster_label",
    n_examples: int = 10,
    selection: str = "random",  # or nearest
    embedding_field: str = "__embedding",
    prompt: Optional[str] = None,  # jinja2 template
):
    
    """
    Labels the clusters using AI based on the given fields and prompt template.

    .. code-block:: python

        # Example usage:
        ds.ai_label_clusters(fields=['field1', 'field2'], selection='nearest')

    :param fields: List of fields used to identify clusters.
    :param new_column: Optional. Name of the new column where the cluster labels will be stored. Defaults to '__cluster_label'.
    :param n_examples: Optional. Number of examples to consider for labeling. Defaults to 10.
    :param selection: Optional. Strategy to select examples for labeling. Can be 'random' or 'nearest'. Defaults to 'random'.
    :param embedding_field: Optional. Name of the embedding field to be used. Defaults to '__embedding'.
    :param prompt: Optional. Jinja2 template string to be used as the prompt for labeling.
    :return: The modified object with labeled clusters.
    """

    if not prompt:
        # Default Jinja2 template
        prompt = """
        Please identify a single shared topic or theme among the following examples in a few words. It's ok if there are a small minority of examples that don't fit with the theme, but if there's no a clear shared topic or theme, just say "No shared topic or theme".
        
        {% for example in examples %}
            ### Example {{ loop.index }}
            {% for field in fields %}
                - {{ field }}: {{ example[field] }}
            {% endfor %}
        {% endfor %}
        
        Now, state the single topic or theme in 3-10 words, no long lists:

        """.strip()
    template = jinja2.Template(prompt)
    queries = []
    for cluster_id in self.cluster_centers.keys():
        cluster_center = self.cluster_centers[cluster_id]
        cluster = self.dataset.filter(
            lambda x: x["__cluster"] == cluster_id
        ).select_columns(fields + [embedding_field])
        if len(cluster) < n_examples:
            examples = list(cluster.select_columns(fields))
        elif selection == "nearest":
            emb_matrix = np.array(cluster[embedding_field])
            similarities = np.dot(emb_matrix, cluster_center)
            top_k = list(np.argsort(similarities)[::-1][:n_examples])
            examples = list(cluster.select(top_k).select_columns(fields))
        elif selection == "random":
            examples = random.choices(
                list(cluster.select_columns(fields)), k=n_examples
            )
        else:
            raise ValueError(f"Unknown selection method: {selection}")

        prompt = template.render(examples=examples, fields=fields)
        queries.append(prompt)

    responses = run_chat_queries_with_openai(queries, self.openai_api_key)
    self.dataset = self.dataset.map(
        lambda x: {new_column: responses[x["__cluster"]]},
    )
    return self


def get_cluster_info(self, n_neighbors: int = 3, field: str = None):
    """
    Retrieves information regarding clusters and their nearest neighbors.

    .. code-block:: python

        # Example usage:
        ds.get_cluster_info(n_neighbors=5, field='title')

    :param n_neighbors: Number of nearest neighbors to be retrieved for each cluster. Defaults to 3.
    :param field: Optional specific field to be printed from the neighbors.
    :raises ValueError: If cluster() or get_embeddings() methods are not called prior to calling this method.
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
        nn = self.get_nearest_neighbors(emb, k=n_neighbors)
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

    """
    Retrieves duplicates within a specified cluster based on a threshold.

    .. code-block:: python

        # Example usage:
        duplicates = get_duplicates(cluster=ds, threshold=0.8, strategy='nearest')

    :param cluster: The dataset cluster to be analyzed for duplicates.
    :param threshold: Similarity threshold to consider two items as duplicates.
    :param strategy: Strategy to select duplicates, can be 'random', 'nearest', or 'furthest'. Defaults to 'random'.
    :raises ValueError: If an unknown strategy is provided.
    :return: List of identified duplicate items within the cluster.
    """

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
    """
    Tunes the threshold for identifying duplicates within a cluster to achieve a target retention rate.

    .. code-block:: python

        # Example usage:
        threshold = tune_threshold(cluster=ds, target_retention=0.9)

    :param cluster: The dataset cluster to be analyzed.
    :param target_retention: Target retention rate to achieve.
    :param tol: Tolerance for the difference between achieved and target retention rate. Defaults to 0.01.
    :param max_iter: Maximum number of iterations for tuning. Defaults to 30.
    :return: The tuned threshold value.
    """

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
    
    """
    Removes semantic near-duplicates from the dataset based on the target retention rate or a specified threshold.

    .. code-block:: python

        # Example usage:
        ds.semdedup(target_retention=0.85, inplace=False)

    :param target_retention: Optional target retention rate. Defaults to 0.8.
    :param threshold: Optional similarity threshold for identifying duplicates.
    :param inplace: Whether to remove duplicates in-place or return a new object. Defaults to True.
    :return: If inplace is False, returns the modified object without duplicates.
    """
    
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
