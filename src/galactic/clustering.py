import datasets
import networkx as nx
import numpy as np
import random
from typing import Optional, Literal
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans, BisectingKMeans, HDBSCAN
from collections import Counter
import jinja2
from .async_openai import run_chat_queries_with_openai

import logging

logger = logging.getLogger("galactic")


def cluster(
    self,
    n_clusters: Optional[int] = None,
    method: Literal[
        "kmeans", "minibatch_kmeans", "bisecting_kmeans", "hdbscan"
    ] = "kmeans",
    embedding_field: str = "__embedding",
    cluster_field: str = "__cluster",
    overwrite: bool = False,
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
    # check if cluster_field is already set
    if cluster_field in self.dataset.column_names:
        if overwrite:
            logger.warning(
                f"You already have clusters in field {cluster_field}. Since overwrite=True, these will be overwritten."
            )
            self.dataset = self.dataset.select_columns(
                [c for c in self.dataset.column_names if c != cluster_field]
            )
        else:
            raise ValueError(
                f"You already have clusters in field {cluster_field}. If you want to overwrite them, pass overwrite=True. Otherwise, use a different 'cluster_field' to create a new clustering."
            )

    # set cluster ids and centers, in a way that allows multiple clusterings!
    if self.cluster_ids is None:
        self.cluster_ids = {}
    if self.cluster_centers is None:
        self.cluster_centers = {}

    # check if emb dimension is large
    if len(self.dataset[embedding_field][0]) >= 384:
        logger.info(
            "Embedding dimension is large, which is fine! But consider also experimenting with dimensionality reduction before clustering."
        )
    if method != "hdbscan" and n_clusters is None:
        raise ValueError(
            "You must specify the number of clusters with n_clusters argument. If you don't want to set this a priori, try using the hdbscan method, which doesn't require you to set it."
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
        self.dataset = self.dataset.map(
            lambda batch: {
                cluster_field: model.predict(
                    batch[embedding_field],
                )
            },
            batched=True,
            batch_size=kwargs.get("batch_size", 1024),
        )

    elif method in ["kmeans", "bisecting_kmeans"]:
        if method == "kmeans":
            model = KMeans(n_clusters=n_clusters, init="k-means++", n_init=1)
        elif method == "bisecting_kmeans":
            model = BisectingKMeans(
                n_clusters=n_clusters,
                init="k-means++",
                n_init=1,
                bisecting_strategy="largest_cluster",
            )
        arr = np.array(self.dataset[embedding_field])
        labels = model.fit_predict(arr)
        self.cluster_ids[cluster_field] = list(set(labels))
        # cluster centers is a dict of id -> center
        self.cluster_centers[cluster_field] = {
            i: model.cluster_centers_[i] for i in range(n_clusters)
        }
        self.dataset = self.dataset.add_column(cluster_field, labels)

    elif method == "hdbscan":
        min_cluster_size = kwargs.get("min_cluster_size", 25)
        model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=kwargs.get("min_samples", min_cluster_size),
            alpha=kwargs.get("alpha", 1.0),
            cluster_selection_method=kwargs.get(
                "cluster_selection_method", "eom"
            ),
            leaf_size=kwargs.get("leaf_size", 40),
            store_centers="medoid",
        )
        arr = np.array(self.dataset[embedding_field])
        labels = model.fit_predict(arr)
        self.cluster_ids[cluster_field] = list(set(labels))
        # cluster centers is a dict of id -> center
        self.cluster_centers[cluster_field] = {
            i: model.medoids_[i] for i in range(max(self.cluster_ids))
        }
        self.dataset = self.dataset.add_column(cluster_field, labels)
    else:
        raise ValueError(f"Unknown clustering method: {method}")


# preferred to filtering out the cluster, because it will remove the cluster from the cluster_ids list
def remove_cluster(self, cluster_field: str, cluster_id: int):
    """Remove a cluster from the dataset."""
    if cluster_field not in self.dataset.column_names:
        raise ValueError(
            f"Cluster field {cluster_field} not found in dataset."
        )
    else:
        self.dataset = self.dataset.filter(
            lambda x: x[cluster_field] != cluster_id
        )
        self.cluster_ids[cluster_field].remove(cluster_id)
        del self.cluster_centers[cluster_field][cluster_id]
    logger.info(
        f"Removed cluster {cluster_id} from clustering {cluster_field} from dataset."
    )


# this does loop through the whole dataset, but that's faster than looping through it N times for N clusters
def _get_clusters(self, cluster_field: str):
    clusters = {}
    self.dataset.map(
        lambda x: clusters.setdefault(x[cluster_field], []).append(x),
    )
    return {
        cluster_id: datasets.Dataset.from_list(cluster)
        for cluster_id, cluster in clusters.items()
    }


def _get_nearest_in_cluster(
    cluster: datasets.Dataset,
    embedding_field: str,
    query: np.ndarray,
    k: int = 3,
):
    """Get the nearest neighbors of a query point in a cluster."""
    emb_matrix = np.array(cluster[embedding_field])
    similarities = np.dot(emb_matrix, query)
    top_k = list(np.argsort(similarities)[::-1][:k])
    examples = list(cluster.select(top_k))
    return examples


def ai_label_clusters(
    self,
    new_column: str,
    context_fields: list[str],
    cluster_field: str = "__cluster",
    embedding_field: str = "__embedding",
    n_examples: int = 10,
    selection: Literal["random", "nearest"] = "random",
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
        prompt = (
            "Please identify a single shared topic or theme among the following examples in a few words. "
            "It's ok if there are a small minority of examples that don't fit with the theme, but if "
            "there's no clear shared topic or theme, just say 'No shared topic or theme'.\n\n"
            "{% for example in examples %}\n"
            "\t### Example {{ loop.index }}\n"
            "\t{% for field in fields %}\n"
            "\t\t- {{ field }}: {{ example[field] }}\n"
            "\t{% endfor %}\n"
            "{% endfor %}\n\n"
            "Now, state the single topic or theme in 3-10 words, no long lists:"
        )
    template = jinja2.Template(prompt)
    queries = []

    logger.info("Splitting dataset into clusters... (this might take a bit).")
    clusters = self._get_clusters(cluster_field)
    ids = []
    for id, cluster in clusters.items():
        ids.append(id)
        cluster_center = self.cluster_centers[cluster_field][id]
        cluster = cluster.select_columns(context_fields + [embedding_field])
        if len(cluster) < n_examples:
            examples = list(cluster.select_columns(context_fields))
        elif selection == "nearest":
            examples = _get_nearest_in_cluster(
                cluster, embedding_field, cluster_center, k=n_examples
            ).select_columns(context_fields)
        elif selection == "random":
            examples = random.choices(
                list(cluster.select_columns(context_fields)), k=n_examples
            )
        else:
            raise ValueError(f"Unknown selection method: {selection}")

        prompt = template.render(examples=examples, fields=context_fields)
        queries.append(prompt)

    responses = run_chat_queries_with_openai(queries, self.openai_api_key)
    id2label = {id: response for id, response in zip(ids, responses)}
    self.dataset = self.dataset.map(
        lambda x: {new_column: id2label[x[cluster_field]]}
    )
    return self


def get_cluster_info(
    self,
    n_neighbors: int = 3,
    cluster_field: str = "__cluster",
    embedding_field: str = "__embedding",
    context_fields: list[str] = [],
    truncate_fields: int = 250,
    verbose: bool = True,
):
    """
    Retrieves information regarding clusters and their nearest neighbors.

    .. code-block:: python

        # Example usage:
        ds.get_cluster_info(n_neighbors=5, field='title')

    :param n_neighbors: Number of nearest neighbors to be retrieved for each cluster. Defaults to 3.
    :param field: Optional specific field to be printed from the neighbors.
    :raises ValueError: If cluster() or get_embeddings() methods are not called prior to calling this method.
    """
    result = []
    if (
        not hasattr(self, "cluster_centers")
        or len(self.cluster_centers.keys()) == 0
    ):
        raise ValueError(
            "You must call cluster() before calling get_cluster_info()"
        )
    if not hasattr(self, "model"):
        raise ValueError(
            "You must call get_embeddings() before calling get_cluster_info(). If your dataset already has embeddings, you need to re-initialize the embedding model with initialize_embedding_model() so that new embeddings can be computed for queries."
        )
    if cluster_field not in self.dataset.column_names:
        raise ValueError(
            f"Cluster field {cluster_field} not found in dataset."
        )

    counts = Counter(self.dataset[cluster_field])

    # for each one, get the 3 nearest neighbors
    clusters = self._get_clusters(cluster_field)
    for id, cluster in clusters.items():
        cluster_center = self.cluster_centers[cluster_field][id]
        examples = _get_nearest_in_cluster(
            cluster, embedding_field, cluster_center, k=n_neighbors
        )

        cluster_info = {
            "cluster_id": id,
            "cluster_size": counts[id],
            # keep only fields in context_fields
            "examples": [
                {field: example[field] for field in context_fields}
                for example in examples
            ],
        }
        result.append(cluster_info)
    if verbose:
        for cluster_info in result:
            print(
                f"Cluster {cluster_info['cluster_id']} ({cluster_info['cluster_size']} items)"
            )
            if len(context_fields) > 0:
                for idx, example in enumerate(cluster_info["examples"]):
                    print(f"### Example {idx + 1}")
                    for field in context_fields:
                        print(
                            f"\t- {field}: {example[field][:truncate_fields]}"
                        )
    return result


def get_duplicates(
    cluster: datasets.Dataset,
    cluster_center: np.ndarray,
    threshold: float,
    emb_matrix: Optional[
        np.ndarray
    ] = None,  # pass this in if you already have it
    similarities: Optional[
        np.ndarray
    ] = None,  # pass this in if you already have it
    embedding_field: str = "__embedding",
    dedup_strategy: Literal["random", "nearest", "furthest"] = "random",
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
    if emb_matrix is None:
        emb_matrix = np.array(cluster[embedding_field])
    if dedup_strategy != "random":
        # centroid = np.mean(emb_matrix, axis=0)
        # no longer doing this because for e.g. hdbscan we might use the medoid instead.
        # center should be passed as an argument
        id2dist = {
            cluster[i]["__id"]: np.dot(emb_matrix[i], cluster_center)
            for i in range(num_points)
        }

    # find connected components
    if similarities is None:
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
        if dedup_strategy == "nearest":
            cmp.sort(key=lambda x: id2dist[x])
        elif dedup_strategy == "furthest":
            cmp.sort(key=lambda x: -id2dist[x])
        duplicates.extend(cmp[1:])
    return duplicates


def tune_threshold(
    cluster: datasets.Dataset,
    cluster_center: np.ndarray,
    target_retention: float,
    tol: float = 0.01,
    max_iter: int = 30,
    embedding_field: str = "__embedding",
    dedup_strategy: Literal["random", "nearest", "furthest"] = "random",
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
    emb_matrix = np.array(cluster[embedding_field])
    similarities = np.dot(emb_matrix, emb_matrix.T)
    min_sim = np.min(similarities)
    max_sim = np.max(similarities)

    # use binary search to find threshold
    print("Minimum similarity:", min_sim, "Maximum similarity:", max_sim)
    lo = min_sim
    hi = max_sim

    for _ in range(max_iter):
        mid = (lo + hi) / 2
        duplicates = get_duplicates(
            cluster=cluster,
            cluster_center=cluster_center,
            threshold=mid,
            emb_matrix=emb_matrix,
            similarities=similarities,
            embedding_field=embedding_field,
            dedup_strategy=dedup_strategy,
        )
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
    embedding_field: str = "__embedding",
    cluster_field: str = "__cluster",
    num_tuning_clusters: int = 3,
    dedup_strategy: Literal["random", "nearest", "furthest"] = "random",
    inplace=True,
):
    """Remove semantic near-duplicates from the dataset."""
    if target_retention is None and threshold is None:
        raise ValueError(
            "You must specify either target_retention or threshold."
        )
    if target_retention is not None and threshold is not None:
        logger.warning(
            f"Both target_retention and threshold specified. Using target_retention to tune threshold on {num_tuning_clusters} clusters."
        )

    # first get all the clusters so we only have to do this once. but don't semdedup noise (-1)
    clusters = self._get_clusters(cluster_field)
    if -1 in clusters:
        del clusters[-1]

    if target_retention is not None:
        tuning_clusters = random.choices(
            list(clusters.keys()), k=num_tuning_clusters
        )
        # tune threshold
        logger.info(f"Tuning threshold on {num_tuning_clusters} clusters...")
        thresholds = []
        for tuning_cluster in tuning_clusters:
            cluster = clusters[tuning_cluster]
            threshold = tune_threshold(
                cluster,
                self.cluster_centers[cluster_field][tuning_cluster],
                target_retention,
                embedding_field=embedding_field,
                dedup_strategy=dedup_strategy,
            )
            thresholds.append(threshold)
        threshold = np.mean(thresholds)
        logger.info(f"Threshold: {round(threshold, 2)}")

    # get duplicates
    remove = []
    for cluster_id, cluster in clusters.items():
        if (
            len(cluster) < 2 or cluster_id < 0
        ):  # don't deduplicate noise or singletons
            continue
        duplicates = get_duplicates(
            cluster,
            self.cluster_centers[cluster_field][cluster_id],
            threshold=threshold,
            embedding_field=embedding_field,
            dedup_strategy=dedup_strategy,
        )
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
