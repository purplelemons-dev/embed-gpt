# mypy: ignore-errors
import numpy as np
from openai import OpenAI
from phase1 import normalize, hashEmbed

embed_to_token = np.load("embed_to_token.npy")
embeddings: np.ndarray = np.load("embeddings.npy")

print(f"{embed_to_token=}")
print(f"{embeddings=}")

client = OpenAI()


def gen_next_best(target:np.ndarray):
    options = {}
    for embed in embeddings:
        hash_val = hashEmbed(embed)
        options[embed_to_token[hash_val]] = 1 - np.dot(target, embed)

    return min(options, key=options.get)

if __name__ == "__main__":
    user_in = input("Enter a string: ")
    print(repr(user_in))
    target = normalize(
        np.array(
            client.embeddings.create(
                model="text-embedding-3-large", input=user_in
            )
            .data[0]
            .embedding,
            dtype=np.float64,
        )
    )
    print(f"Next best token: {gen_next_best(target)}")
