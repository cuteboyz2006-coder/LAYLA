import math
import random


class TransformerDecoder:
    def __init__(
        self,
        embedding_size,
        vocab_size,
        hidden_size=32,
        seed=42
    ):
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        random.seed(seed)

        # Feed-forward weights
        self.w1 = self._matrix(
            embedding_size,
            hidden_size
        )

        self.b1 = [0.0] * hidden_size

        self.w2 = self._matrix(
            hidden_size,
            embedding_size
        )

        self.b2 = [0.0] * embedding_size

        # Output projection:
        # hidden representation -> vocabulary logits
        self.output_weights = self._matrix(
            embedding_size,
            vocab_size
        )

        self.output_bias = [0.0] * vocab_size

    def _matrix(self, rows, cols):
        return [
            [
                random.uniform(-0.1, 0.1)
                for _ in range(cols)
            ]
            for _ in range(rows)
        ]

    def _linear(self, vector, matrix, bias):
        output = []

        for column in range(len(bias)):
            value = bias[column]

            for row in range(len(vector)):
                value += vector[row] * matrix[row][column]

            output.append(value)

        return output

    def _relu(self, vector):
        return [
            max(0.0, value)
            for value in vector
        ]

    def feed_forward(self, vector):
        hidden = self._linear(
            vector,
            self.w1,
            self.b1
        )

        hidden = self._relu(hidden)

        output = self._linear(
            hidden,
            self.w2,
            self.b2
        )

        return output

    def causal_attention(self, embeddings):
        outputs = []

        for position in range(len(embeddings)):
            query = embeddings[position]

            visible = embeddings[:position + 1]

            scores = []

            for key in visible:
                score = sum(
                    query[i] * key[i]
                    for i in range(self.embedding_size)
                )

                score /= math.sqrt(
                    self.embedding_size
                )

                scores.append(score)

            maximum = max(scores)

            exp_scores = [
                math.exp(score - maximum)
                for score in scores
            ]

            total = sum(exp_scores)

            weights = [
                value / total
                for value in exp_scores
            ]

            output = [0.0] * self.embedding_size

            for i, weight in enumerate(weights):
                for dimension in range(
                    self.embedding_size
                ):
                    output[dimension] += (
                        weight * visible[i][dimension]
                    )

            outputs.append(output)

        return outputs

    def forward(self, embeddings):
        if not embeddings:
            return []

        # Causal self-attention
        attention_output = self.causal_attention(
            embeddings
        )

        decoder_output = []

        for vector in attention_output:
            ff_output = self.feed_forward(vector)

            # Simple residual connection
            combined = [
                vector[i] + ff_output[i]
                for i in range(self.embedding_size)
            ]

            decoder_output.append(combined)

        return decoder_output

    def logits(self, decoder_output):
        all_logits = []

        for vector in decoder_output:
            values = self._linear(
                vector,
                self.output_weights,
                self.output_bias
            )

            all_logits.append(values)

        return all_logits

    def predict_next_token(self, logits):
        if not logits:
            return None

        last_logits = logits[-1]

        best_id = 0
        best_value = last_logits[0]

        for token_id in range(
            1,
            len(last_logits)
        ):
            if last_logits[token_id] > best_value:
                best_value = last_logits[token_id]
                best_id = token_id

        return best_id
