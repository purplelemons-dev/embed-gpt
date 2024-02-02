# mypy: ignore-errors
from openai import OpenAI
import numpy as np
import py_dotenv
from tiktoken_ext.openai_public import cl100k_base

py_dotenv.read_dotenv(".env")


def hashEmbed(embed: np.ndarray) -> int:
    return hash(embed.data.hex()) % SIZE


SIZE = 2**25
DIMS = 3072
TEXT_SIZE = 26
client = OpenAI()
cl100k = cl100k_base()
tokens = set(i[:TEXT_SIZE] for i in cl100k["mergeable_ranks"].keys())
tokens.update(b"<|endoftext|>")

embed_to_token: np.ndarray = np.ndarray((SIZE,), dtype=f"|S{TEXT_SIZE}")
embeddings = np.ndarray((len(tokens), DIMS), dtype=np.float64)

for idx, token in enumerate(tokens):
    embedding = np.array(
        client.embeddings.create(
            model="text-embedding-3-large", dimensions=DIMS, input=token
        )
        .data[0]
        .embedding,
        dtype=np.float64,
    )
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding /= norm
    embed_to_token[hashEmbed(embedding)] = token
    embeddings[idx] = embedding


np.save("embed_to_token.npy", embed_to_token)
np.save("embeddings.npy", embeddings)
