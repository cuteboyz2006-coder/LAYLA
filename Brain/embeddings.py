import random


class TokenEmbeddings:
    def __init__(self, vocab_size, embedding_size=16, seed=42):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        random.seed(seed)

        self.weights = []

        for _ in range(vocab_size):
            vector = [
                random.uniform(-0.1, 0.1)
                for _ in range(embedding_size)
            ]

            self.weights.append(vector)

    def get(self, token_id):
        if token_id < 0 or token_id >= self.vocab_size:
            raise ValueError("Invalid token ID")

        return self.weights[token_id]

    def encode(self, token_ids):
        return [
            self.get(token_id)
            for token_id in token_ids
        ]
