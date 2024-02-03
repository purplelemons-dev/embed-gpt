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

cl100k = cl100k_base()

tokens: set[bytes] = set(i[:TEXT_SIZE] for i in cl100k["mergeable_ranks"].keys())
tokens.update(b"<|endoftext|>")

def normalize(embed: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(embed)
    if norm > 0:
        embed /= norm
    return embed

if __name__ == "__main__":

    client = OpenAI()

    embed_to_token: np.ndarray = np.ndarray((SIZE,), dtype=f"|S{TEXT_SIZE}")
    embeddings = np.ndarray((len(tokens), DIMS), dtype=np.float64)

    intermediate:list[list[str]] = []
    current = []
    for i in tokens:
        if len(current) >= 1024:
            intermediate.append(current)
            current = []
        current.append(str(i))
    intermediate.append(current)

    amount_done = 0

    for idx, token_list in enumerate(intermediate):
        print(amount_done)
        for token, embedding_data in zip(token_list, client.embeddings.create(
                model="text-embedding-3-large", dimensions=DIMS, input=token_list
            )
            .data):
            embedding = np.array(embedding_data.embedding, dtype=np.float64)
            embedding = normalize(embedding)
            embed_to_token[hashEmbed(embedding)] = token
            embeddings[amount_done] = embedding
            amount_done += 1


    np.save("embed_to_token.npy", embed_to_token)
    np.save("embeddings.npy", embeddings)
