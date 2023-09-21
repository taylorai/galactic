import re
import unicodedata
from datasketch import MinHash, LeanMinHash

from logging import getLogger

logger = getLogger("galactic")


def replace_whitespace_with_underscore(text: str):
    return re.sub(r"\s+", "_", text)


def compute_minhash(text: str, k: int = 9, num_perm: int = 128) -> LeanMinHash:
    # first normalize
    text = unicodedata.normalize("NFKC", text)

    # then replace whitespace with underscore
    text = replace_whitespace_with_underscore(text)

    # shingle
    shingles = set()
    for i in range(len(text) - k + 1):
        shingles.add(text[i : i + k])

    # compute minhash
    m = MinHash(num_perm=num_perm)
    for s in shingles:
        m.update(s.encode("utf8"))
    return LeanMinHash(m)


def compute_minhashes(self, field: str, k: int = 9, num_perm: int = 128):
    # make sure field exists
    if field not in self.dataset.features:
        raise ValueError(f"Field {field} not found in dataset.")

    def minhash_(sample):
        m = compute_minhash(sample[field], k=k, num_perm=num_perm)
        buf = bytearray(m.bytesize())
        m.serialize(buf)
        return {f"__minhash__{field}": buf}

    self.dataset = self.dataset.map(minhash_)
    logger.info(
        f"Computed minhashes for field {field}, added minhash metadata to '__minhash__{field}'."
    )
    return self
