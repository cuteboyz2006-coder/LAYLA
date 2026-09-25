import math


class SelfAttention:
    def __init__(self, embedding_size):
        self.embedding_size = embedding_size

    def dot(self, a, b):
        return sum(x * y for x, y in zip(a, b))

    def softmax(self, values):
        maximum = max(values)

        exp_values = [
            math.exp(value - maximum)
            for value in values
        ]

        total = sum(exp_values)

        return [
            value / total
            for value in exp_values
        ]

    def forward(self, embeddings):
        if not embeddings:
            return []

        outputs = []

        for query in embeddings:
            scores = []

            for key in embeddings:
                score = self.dot(query, key)

                score /= math.sqrt(self.embedding_size)

                scores.append(score)

            weights = self.softmax(scores)

            output = []

            for dimension in range(self.embedding_size):
                value = sum(
                    weights[i] * embeddings[i][dimension]
                    for i in range(len(embeddings))
                )

                output.append(value)

            outputs.append(output)

        return outputs
