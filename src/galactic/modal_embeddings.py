# if you want to use Modal Labs for embeddings, you'll need to deploy this in YOUR modal account.
# then, if you're authenticated to Modal, you should be able to look up the function and use it as
# an embeddings backend.
from modal import Image, Stub, method
from typing import Any, Union
from pathlib import Path

image = (
    # Python 3.11+ not yet supported for torch.compile
    Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "onnx==1.14.1",
        "onnxruntime==1.15.1",
        "numpy==1.23.4",
        "transformers>=4.30.2",
        "hf-transfer~=0.1",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

stub = Stub(name="gte-small-cpu", image=image)


@stub.cls()
class GTE:
    def __enter__(self):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        model_path = hf_hub_download(
            "TaylorAI/galactic-models",
            filename="model_quantized.onnx",
        )
        tokenizer_path = "Supabase/gte-small"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.session = ort.InferenceSession(model_path)

    @method()
    def forward(self, texts: Union[str, list[str]]):
        if isinstance(texts, str):
            texts = [texts]
        import numpy as np

        max_length = 512
        outputs = []
        for text in texts:
            tokenized = self.tokenizer(text, return_tensors="np")

            # pad to be a multiple of max_length
            rounded_len = int(
                np.ceil(len(tokenized["input_ids"][0]) / max_length)
                * max_length
            )
            for k in tokenized:
                tokenized[k] = np.pad(
                    tokenized[k],
                    ((0, 0), (0, rounded_len - len(tokenized[k][0]))),
                    constant_values=0,
                )

            # reshape into batch with max length
            for k in tokenized:
                tokenized[k] = tokenized[k].reshape((-1, max_length))
            outs = []
            for seq in range(len(tokenized["input_ids"])):
                out = self.session.run(
                    None,
                    {
                        "input_ids": tokenized["input_ids"][seq : seq + 1],
                        "attention_mask": tokenized["attention_mask"][
                            seq : seq + 1
                        ],
                        "token_type_ids": tokenized["token_type_ids"][
                            seq : seq + 1
                        ],
                    },
                )[0]
                outs.append(out)
            out = np.concatenate(outs, axis=0)  # bsz, seq_len, hidden_size

            # average over seq_len and bsz to get one embedding
            avg = np.mean(out, axis=(0, 1))

            # norm
            outputs.append((avg / np.linalg.norm(avg)).tolist())

        return outputs
