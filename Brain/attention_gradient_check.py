
import math


class AttentionGradientChecker:

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

    def causal_attention(
        self,
        embeddings,
        embedding_size
    ):
        last_position = len(embeddings) - 1

        visible = embeddings[:last_position + 1]
        query = embeddings[last_position]

        scale = math.sqrt(embedding_size)

        scores = []

        for key in visible:
            score = 0.0

            for dimension in range(embedding_size):
                score += (
                    query[dimension]
                    * key[dimension]
                )

            score /= scale
            scores.append(score)

        weights = self.softmax(scores)

        output = [
            0.0
            for _ in range(embedding_size)
        ]

        for index, weight in enumerate(weights):
            for dimension in range(embedding_size):
                output[dimension] += (
                    weight
                    * visible[index][dimension]
                )

        return output

    def objective(
        self,
        embeddings,
        embedding_size,
        output_gradient
    ):
        output = self.causal_attention(
            embeddings,
            embedding_size
        )

        return sum(
            output[i] * output_gradient[i]
            for i in range(embedding_size)
        )

    def numerical_gradient(
        self,
        embeddings,
        embedding_size,
        output_gradient,
        position,
        dimension,
        epsilon=0.000001
    ):
        original = embeddings[position][dimension]

        embeddings[position][dimension] = (
            original + epsilon
        )

        loss_plus = self.objective(
            embeddings,
            embedding_size,
            output_gradient
        )

        embeddings[position][dimension] = (
            original - epsilon
        )

        loss_minus = self.objective(
            embeddings,
            embedding_size,
            output_gradient
        )

        embeddings[position][dimension] = original

        return (
            loss_plus - loss_minus
        ) / (
            2.0 * epsilon
        )

    def check(
        self,
        embeddings,
        embedding_size,
        output_gradient
    ):
        from Brain.attention_backprop import (
            AttentionBackprop
        )

        backprop = AttentionBackprop()

        analytical_gradients = (
            backprop.calculate_input_gradient(
                embeddings=embeddings,
                output_gradient=output_gradient,
                embedding_size=embedding_size
            )
        )

        results = []

        for position in range(len(embeddings)):
            for dimension in range(embedding_size):

                analytical = (
                    analytical_gradients[position][dimension]
                )

                numerical = (
                    self.numerical_gradient(
                        embeddings=embeddings,
                        embedding_size=embedding_size,
                        output_gradient=output_gradient,
                        position=position,
                        dimension=dimension
                    )
                )

                difference = abs(
                    analytical - numerical
                )

                results.append({
                    "position": position,
                    "dimension": dimension,
                    "analytical": analytical,
                    "numerical": numerical,
                    "difference": difference
                })

        return results
