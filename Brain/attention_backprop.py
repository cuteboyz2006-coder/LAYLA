import math


class AttentionBackprop:

    def softmax(self, values):
        if not values:
            return []

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

        if not output_gradient:
            return []

        last_position = len(embeddings) - 1

        visible = embeddings[
            :last_position + 1
        ]

        query = embeddings[last_position]

        # --------------------------------
        # Forward attention scores
        # --------------------------------

        scale = math.sqrt(
            embedding_size
        )

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

            score /= scale

            scores.append(score)

        # --------------------------------
        # Attention weights
        # --------------------------------

        weights = self.softmax(scores)

        # --------------------------------
        # Gradient containers
        # --------------------------------

        input_gradients = [
            [0.0 for _ in range(embedding_size)]
            for _ in range(len(embeddings))
        ]

        # --------------------------------
        # 1. Gradient through VALUE path
        #
        # output =
        # sum(weight * value)
        # --------------------------------

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

        # --------------------------------
        # 2. Gradient wrt attention weights
        # --------------------------------

        weight_gradients = []

        for position in range(
            len(visible)
        ):

            gradient = 0.0

            for dimension in range(
                embedding_size
            ):

                gradient += (
                    output_gradient[dimension]
                    * visible[position][dimension]
                )

            weight_gradients.append(
                gradient
            )

        # --------------------------------
        # 3. Softmax backward
        # --------------------------------

        score_gradients = [
            0.0
            for _ in range(len(weights))
        ]

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

            score_gradients[i] = gradient

        # --------------------------------
        # 4. Gradient through dot-product
        #
        # score = dot(query, key) / sqrt(d)
        # --------------------------------

        for position in range(
            len(visible)
        ):

            score_gradient = (
                score_gradients[position]
            )

            key = visible[position]

            # Gradient wrt key
            for dimension in range(
                embedding_size
            ):

                input_gradients[position][dimension] += (
                    score_gradient
                    * query[dimension]
                    / scale
                )

            # Gradient wrt query
            for dimension in range(
                embedding_size
            ):

                input_gradients[last_position][dimension] += (
                    score_gradient
                    * key[dimension]
                    / scale
                )

        return input_gradients
