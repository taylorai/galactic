### NOTE: THIS FILE DOESNT DO ANYTHING RIGHT NOW.
# STILL FIGURING OUT A CUSTOM SAVING FORMAT THAT MAKES SENSE.

import gzip
import tarfile
import json
import os


def save_to_disk(self, path: str, metadata: dict):
    """
    Save the GalacticDataset to disk as a GZipped TAR archive.

    Parameters:
    - path (str): The path where the archive will be saved.
    - metadata (dict): A dictionary containing metadata.

    Returns:
    - None
    """

    # Create temporary files for data and metadata
    data_file = "data.jsonl"
    metadata_file = "metadata.json"

    # Write the dataset to a JSONL file
    with open(data_file, "w") as f:
        for sample in self:
            json.dump(sample, f)
            f.write("\n")

    # Write metadata to a JSON file
    with open(metadata_file, "w") as f:
        json.dump(metadata, f)

    # Create a compressed TAR archive
    with tarfile.open(path, "w:gz") as tar:
        tar.add(data_file, arcname="data.jsonl")
        tar.add(metadata_file, arcname="metadata.json")

    # Remove temporary files
    os.remove(data_file)
    os.remove(metadata_file)
