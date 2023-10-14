# if you want to use Modal Labs for embeddings, you'll need to deploy this in YOUR modal account under the name 'modal_embeddings'.
# then, if you're authenticated to Modal, you should be able to look up the function and use it as an embeddings backend.
from modal import Image, Stub, method, Cls
from typing import Union, Literal

image = (
    # Python 3.11+ not yet supported for torch.compile
    Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch",
        "galactic-ai>=0.2.15",
        "huggingface-hub",
        "hf-transfer~=0.1",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

stub = Stub(name="modal_embeddings", image=image)


# Embedding model that runs on Modal, wrapping the ONNX embedding model.
@stub.cls()
class RemoteEmbeddingModel:
    def __enter__(self):
        from huggingface_hub import hf_hub_download

        model_registry = {
            "gte-small": {
                "remote_file": "model_quantized.onnx",
                "tokenizer": "Supabase/gte-small",
            },
            "gte-tiny": {
                "remote_file": "gte-tiny.onnx",
                "tokenizer": "Supabase/gte-small",
            },
            "bge-micro": {
                "remote_file": "bge-micro.onnx",
                "tokenizer": "BAAI/bge-small-en-v1.5",
            },
            "bge-micro-v2": {
                "remote_file": "bge-micro-v2.onnx",
                "tokenizer": "BAAI/bge-small-en-v1.5",
            },
        }

        # download allll the things!
        for model_name in model_registry:
            hf_hub_download(
                "TaylorAI/galactic-models",
                filename=model_registry[model_name]["remote_file"],
            )

    def __init__(self, model_name="bge-micro"):
        from galactic.embedding_backends.onnx_backend import ONNXEmbeddingModel

        self.model = ONNXEmbeddingModel(
            model_name=model_name,
        )

    @method()
    def embed(
        self,
        text: str,
        normalize: bool = False,
        pad: bool = False,
        split_strategy: Literal["truncate", "greedy", "even"] = "even",
    ):
        return self.model.embed(
            text, normalize=normalize, pad=pad, split_strategy=split_strategy
        )

    @method()
    def embed_batch(
        self,
        texts: list[str],
        normalize: bool = False,
        pad: bool = False,
        split_strategy: Literal["truncate", "greedy", "even"] = "even",
    ):
        return self.model.embed_batch(
            texts, normalize=normalize, pad=pad, split_strategy=split_strategy
        )
