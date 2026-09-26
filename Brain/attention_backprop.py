import math


class AttentionBackprop:

    def softmax(self, values):
        maximum = max(values)

        exp_values = [
            math.exp(value - maximum)
            for value in values
        ]

        total = sum(exp_values)

        if total == 0:
            return [0.0] * len(values)

        return [
            value / total
            for value in exp_values
        ]

    def calculate_input_gradient(
        self,
        embeddings,
        output_gradient,
        embedding_size
    ):
        if not embeddings:
            return []

        last_position = len(embeddings) - 1

        query = embeddings[last_position]

        visible = embeddings[
            :last_position + 1
        ]

        # -------------------------
        # Attention scores
        # -------------------------

        scores = []

        for key in visible:

            score = 0.0

            for dimension in range(
                embedding_size
            ):
                score += (
                    query[dimension]
                    * key[dimension]
                )

            score /= math.sqrt(
                embedding_size
            )

            scores.append(score)

        # -------------------------
        # Attention weights
        # -------------------------

        weights = self.softmax(
            scores
        )

        # -------------------------
        # Initial gradients
        # -------------------------

        input_gradients = [
            [0.0 for _ in range(embedding_size)]
            for _ in range(len(embeddings))
        ]

        # -------------------------
        # Value-path gradient
        # -------------------------

        for position in range(
            len(visible)
        ):

            weight = weights[position]

            for dimension in range(
                embedding_size
            ):
                input_gradients[position][dimension] += (
                    weight
                    * output_gradient[dimension]
                )

        # -------------------------
        # Attention-weight gradient
        # -------------------------

        weight_gradients = []

        for position in range(
            len(visible)
        ):

            value = 0.0

            for dimension in range(
                embedding_size
            ):
                value += (
                    output_gradient[dimension]
                    * visible[position][dimension]
                )

            weight_gradients.append(
                value
            )

        # -------------------------
        # Softmax gradient
        # -------------------------

        score_gradients = []

        for i in range(
            len(weights)
        ):

            gradient = 0.0

            for j in range(
                len(weights)
            ):

                if i == j:
                    jacobian = (
                        weights[i]
                        * (1.0 - weights[i])
                    )

                else:
                    jacobian = (
                        -weights[i]
                        * weights[j]
                    )

                gradient += (
                    weight_gradients[j]
                    * jacobian
                )

            score_gradients.append(
                gradient
            )

        # -------------------------
        # Score gradient
        #
        # score = query · key / sqrt(d)
        # -------------------------

        scale = math.sqrt(
            embedding_size
        )

        # Gradient to query
        for dimension in range(
            embedding_size
        ):

            value = 0.0

            for position in range(
                len(visible)
            ):
                value += (
                    score_gradients[position]
                    * visible[position][dimension]
                    / scale
                )

            input_gradients[last_position][dimension] += (
                value
            )

        # -------------------------
        # Gradient to keys
        # -------------------------

        for position in range(
            len(visible)
        ):

            for dimension in range(
                embedding_size
            ):

                input_gradients[position][dimension] += (
                    score_gradients[position]
                    * query[dimension]
                    / scale
                )

        return input_gradients
