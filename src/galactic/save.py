import gzip
import tarfile
import json
import os
import csv

class GalacticDataset:
    # Dummy dataset class for the sake of demonstration
    def __iter__(self):
        # This should yield samples from your dataset
        yield {"id": 1, "name": "sample1"}
        yield {"id": 2, "name": "sample2"}

def save_to_disk(dataset, path: str, metadata: dict, format: str = "tar"):
    """
    Save the GalacticDataset to disk in the specified format.

    Parameters:
    - dataset: The dataset object to be saved.
    - path (str): The path where the file will be saved.
    - metadata (dict): A dictionary containing metadata.
    - format (str): Desired format ("jsonl", "csv", "tar").

    Returns:
    - None
    """
    
    if format == "jsonl":
        # Write the dataset to a JSONL file
        with open(path, "w") as f:
            for sample in dataset:
                json.dump(sample, f)
                f.write("\n")
                
    elif format == "csv":
        # Assuming the dataset samples are dictionaries with consistent keys
        fieldnames = dataset[0].keys()
        with open(path, "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for sample in dataset:
                writer.writerow(sample)
                
    elif format == "tar":
        # Similar to your original code but with minor improvements
        data_file = "data.jsonl"
        metadata_file = "metadata.json"

        # Write the dataset to a JSONL file
        with open(data_file, "w") as f:
            for sample in dataset:
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
        
    else:
        raise ValueError(f"Unsupported format: {format}")
