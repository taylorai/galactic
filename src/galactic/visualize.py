from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def plot_embeddings(
    self,
    embedding_field: str,
    color_by: Optional[str] = None,
    save_path: Optional[str] = None,
):
    # Check the dimensionality
    if len(self.dataset[embedding_field][0]) != 2:
        raise ValueError("The dimensionality of the embeddings must be 2.")

    # Separate X and Y coordinates
    x_coords, y_coords = zip(*self.dataset[embedding_field])

    # Check if color_by is supplied
    if color_by:
        labels = self.dataset[color_by]
        unique_labels = list(set(labels))

        cmap = ListedColormap(
            plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        )

        for i, label in enumerate(unique_labels):
            idx = [j for j, x in enumerate(labels) if x == label]
            plt.scatter(
                np.array(x_coords)[idx],
                np.array(y_coords)[idx],
                color=cmap(i),
                label=label,
                s=1,
            )

        plt.legend(title=color_by)

    else:
        plt.scatter(x_coords, y_coords, s=1)

    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.title(f"2D {embedding_field} embeddings")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
