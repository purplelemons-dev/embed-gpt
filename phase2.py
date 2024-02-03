# mypy: ignore-errors
import numpy as np
from openai import OpenAI
from phase1 import normalize, hashEmbed

embed_to_token = np.load("embed_to_token.npy")
embeddings: np.ndarray = np.load("embeddings.npy")

client = OpenAI()


def gen_next_best(target:np.ndarray):
    options = {}
    for embed in embeddings:
        hash_val = hashEmbed(embed)
        options[embed_to_token[hash_val]] = 1 - np.dot(target, embed)

    return str(min(options, key=options.get)).replace('b"b\'',"").replace('\'"','')

if __name__ == "__main__":
    while True:
        user_in = input("Enter a string: ")
        out = user_in
        while not out.endswith("<|endoftext|>"):
            print(out)
            out += gen_next_best(client.embeddings.create(
                model="text-embedding-3-large", input=out
            ).data[0].embedding)
        print(out)
