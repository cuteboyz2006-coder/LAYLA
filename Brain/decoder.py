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

        # --------------------------------
        # Feed-forward network
        # --------------------------------

        self.w1 = self._matrix(
            embedding_size,
            hidden_size
        )

        self.b1 = [
            0.0
            for _ in range(hidden_size)
        ]

        self.w2 = self._matrix(
            hidden_size,
            embedding_size
        )

        self.b2 = [
            0.0
            for _ in range(embedding_size)
        ]

        # --------------------------------
        # Output projection
        # --------------------------------

        self.output_weights = self._matrix(
            embedding_size,
            vocab_size
        )

        self.output_bias = [
            0.0
            for _ in range(vocab_size)
        ]

    # ====================================
    # Matrix initialization
    # ====================================

    def _matrix(self, rows, cols):

        return [
            [
                random.uniform(-0.05, 0.05)
                for _ in range(cols)
            ]
            for _ in range(rows)
        ]

    # ====================================
    # Linear layer
    # ====================================

    def _linear(
        self,
        vector,
        matrix,
        bias
    ):

        output = []

        for column in range(
            len(bias)
        ):

            value = bias[column]

            for row in range(
                len(vector)
            ):

                value += (
                    vector[row]
                    * matrix[row][column]
                )

            output.append(value)

        return output

    # ====================================
    # ReLU
    # ====================================

    def _relu(
        self,
        vector
    ):

        return [
            max(0.0, value)
            for value in vector
        ]

    # ====================================
    # Feed Forward
    # ====================================

    def feed_forward(
        self,
        vector
    ):

        hidden = self._linear(
            vector,
            self.w1,
            self.b1
        )

        hidden = self._relu(
            hidden
        )

        output = self._linear(
            hidden,
            self.w2,
            self.b2
        )

        return output

    # ====================================
    # Causal Self Attention
    # ====================================

    def causal_attention(
        self,
        embeddings
    ):

        if not embeddings:
            return []

        outputs = []

        scale = math.sqrt(
            self.embedding_size
        )

        for position in range(
            len(embeddings)
        ):

            query = embeddings[position]

            # Only previous/current tokens
            # are visible.

            visible = embeddings[
                :position + 1
            ]

            scores = []

            # ----------------------------
            # Attention scores
            # ----------------------------

            for key in visible:

                score = 0.0

                for dimension in range(
                    self.embedding_size
                ):

                    score += (
                        query[dimension]
                        * key[dimension]
                    )

                score /= scale

                scores.append(
                    score
                )

            # ----------------------------
            # Stable softmax
            # ----------------------------

            maximum = max(
                scores
            )

            exp_scores = [
                math.exp(
                    score - maximum
                )
                for score in scores
            ]

            total = sum(
                exp_scores
            )

            if total == 0:
                weights = [
                    1.0 / len(exp_scores)
                    for _ in exp_scores
                ]
            else:
                weights = [
                    value / total
                    for value in exp_scores
                ]

            # ----------------------------
            # Weighted values
            # ----------------------------

            output = [
                0.0
                for _ in range(
                    self.embedding_size
                )
            ]

            for index, weight in enumerate(
                weights
            ):

                value = visible[index]

                for dimension in range(
                    self.embedding_size
                ):

                    output[dimension] += (
                        weight
                        * value[dimension]
                    )

            outputs.append(
                output
            )

        return outputs

    # ====================================
    # Decoder forward
    # ====================================

    def forward(
        self,
        embeddings
    ):

        if not embeddings:
            return []

        attention_output = (
            self.causal_attention(
                embeddings
            )
        )

        decoder_output = []

        for vector in attention_output:

            ff_output = (
                self.feed_forward(
                    vector
                )
            )

            # Residual connection

            combined = [
                vector[i]
                + ff_output[i]
                for i in range(
                    self.embedding_size
                )
            ]

            decoder_output.append(
                combined
            )

        return decoder_output

    # ====================================
    # Vocabulary logits
    # ====================================

    def logits(
        self,
        decoder_output
    ):

        all_logits = []

        for vector in decoder_output:

            values = self._linear(
                vector,
                self.output_weights,
                self.output_bias
            )

            all_logits.append(
                values
            )

        return all_logits

    # ====================================
    # Greedy next-token prediction
    # ====================================

    def predict_next_token(
        self,
        logits
    ):

        if not logits:
            return None

        last_logits = logits[-1]

        if not last_logits:
            return None

        best_id = 0
        best_value = last_logits[0]

        for token_id in range(
            1,
            len(last_logits)
        ):

            if (
                last_logits[token_id]
                > best_value
            ):

                best_value = (
                    last_logits[token_id]
                )

                best_id = token_id

        return best_id
