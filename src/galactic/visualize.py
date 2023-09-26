import re
from typing import Optional, Literal
import matplotlib.pyplot as plt
import seaborn as sns

# from matplotlib.colors import ListedColormap
import numpy as np

import logging

logger = logging.getLogger("galactic")


def plot_embeddings(
    self,
    embedding_field: str,
    color_by: Optional[str] = None,
    save_path: Optional[str] = None,
    dot_size: int = 2,
    theme: Literal[
        "darkgrid", "whitegrid", "dark", "white", "ticks"
    ] = "whitegrid",
    width: Optional[int] = None,
    height: Optional[int] = None,
):
    # Check the dimensionality
    if len(self.dataset[embedding_field][0]) != 2:
        raise ValueError("The dimensionality of the embeddings must be 2.")

    # Separate X and Y coordinates
    x_coords, y_coords = zip(*self.dataset[embedding_field])

    # if width and height provided, set them
    if width and height:
        plt.figure(figsize=(width, height))

    sns.set(style=theme)

    if color_by:
        labels = self.dataset[color_by]
        unique_labels = list(set(labels))
        palette = sns.color_palette("husl", len(unique_labels))

        for i, label in enumerate(unique_labels):
            idx = [j for j, x in enumerate(labels) if x == label]
            sns.scatterplot(
                x=np.array(x_coords)[idx],
                y=np.array(y_coords)[idx],
                color=palette[i],
                label=label,
                s=dot_size,
            )

        plt.legend(title=color_by, fontsize="smaller", markerscale=2)
    else:
        sns.scatterplot(x=x_coords, y=y_coords, s=dot_size)

    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    escaped_color_by = re.sub(r"([#\$%&~_^\\])", r"\\\1", color_by)
    plt.title(
        "2D embeddings, colored by  $\\mathtt{" + escaped_color_by + "}$"
    )

    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved plot to {save_path}")
    else:
        plt.show()
