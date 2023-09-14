import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
import ctranslate2

class EmbeddingModel:
    def __init__(self, model_path, tokenizer_path, model_type, max_length=512):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.max_length = max_length
        self.model_type = model_type
        if model_type == "onnx":
            self.session = ort.InferenceSession(self.model_path)
        elif model_type == "ctranslate2":
            self.session = ctranslate2.Encoder(self.model_path, compute_type="int8")

    def split_and_tokenize(self, text):
        # first make into tokens
        tokenized = self.tokenizer(text, return_tensors="np")

        # pad to be a multiple of max_length
        rounded_len = int(np.ceil(len(tokenized["input_ids"][0]) / self.max_length) * self.max_length)
        for k in tokenized:
            tokenized[k] = np.pad(tokenized[k], ((0, 0), (0, rounded_len - len(tokenized[k][0]))), constant_values=0)

        # reshape into batch with max length
        for k in tokenized:
            tokenized[k] = tokenized[k].reshape((-1, self.max_length))

        return tokenized

    def forward_onnx(self, input):
        outs = []
        for seq in range(len(input["input_ids"])):
            out = self.session.run(
                None, {
                    "input_ids": input["input_ids"][seq:seq+1],
                    "attention_mask": input["attention_mask"][seq:seq+1],
                    "token_type_ids": input["token_type_ids"][seq:seq+1]
                }
            )[0]
            outs.append(out)
        out = np.concatenate(outs, axis=0) # bsz, seq_len, hidden_size
        return out
    
    def forward_ctranslate2(self, input):
        input_ids = input["input_ids"].tolist()
        out = self.session.forward_batch(input_ids).pooler_output
        return out

    def predict(self, input):
        input = self.split_and_tokenize(input)
        if self.model_type == "onnx":
            # print("onnx")
            out = self.forward_onnx(input)
            # print("shape", out.shape)
            out = np.mean(out, axis=(0, 1)) # mean of seq and bsz
        elif self.model_type == "ctranslate2":
            # print("ct2")
            out = self.forward_ctranslate2(input)
            out =  np.mean(out, axis=0) # mean just over bsz
        # normalize to unit vector
        return out / np.linalg.norm(out)

    def __call__(self, input):
        return self.predict(input)