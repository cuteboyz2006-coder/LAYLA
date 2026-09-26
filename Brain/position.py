import math


class PositionalEncoding:
    def __init__(self, embedding_size):
        self.embedding_size = embedding_size

    def encode(self, length):
        positions = []

        for position in range(length):

            vector = []

            for dimension in range(
                self.embedding_size
            ):

                angle = (
                    position
                    / math.pow(
                        10000,
                        (
                            2
                            * (dimension // 2)
                            / self.embedding_size
                        )
                    )
                )

                if dimension % 2 == 0:
                    value = math.sin(angle)
                else:
                    value = math.cos(angle)

                vector.append(value)

            positions.append(vector)

        return positions

    def add(
        self,
        embeddings
    ):
        if not embeddings:
            return []

        positional_vectors = self.encode(
            len(embeddings)
        )

        output = []

        for position in range(
            len(embeddings)
        ):

            vector = []

            for dimension in range(
                self.embedding_size
            ):

                value = (
                    embeddings[position][dimension]
                    + positional_vectors[position][dimension]
                )

                vector.append(value)

            output.append(vector)

        return output
