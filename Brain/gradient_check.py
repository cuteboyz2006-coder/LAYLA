import math


class GradientChecker:

    def softmax(self, logits):
        maximum = max(logits)

        exp_values = [
            math.exp(
                value - maximum
            )
            for value in logits
        ]

        total = sum(exp_values)

        return [
            value / total
            for value in exp_values
        ]

    def loss(
        self,
        logits,
        target_id
    ):
        probabilities = self.softmax(
            logits
        )

        probability = max(
            probabilities[target_id],
            1e-12
        )

        return -math.log(
            probability
        )

    def analytical_output_gradient(
        self,
        decoder_vector,
        output_weights,
        output_bias,
        target_id
    ):
        vocab_size = len(
            output_bias
        )

        logits = []

        for token_id in range(
            vocab_size
        ):

            value = output_bias[
                token_id
            ]

            for dimension in range(
                len(decoder_vector)
            ):

                value += (
                    decoder_vector[dimension]
                    * output_weights[dimension][token_id]
                )

            logits.append(value)

        probabilities = self.softmax(
            logits
        )

        probabilities[target_id] -= 1.0

        return (
            decoder_vector,
            probabilities
        )

    def numerical_gradient(
        self,
        decoder_vector,
        output_weights,
        output_bias,
        target_id,
        row,
        column,
        epsilon=0.000001
    ):

        original = (
            output_weights[row][column]
        )

        # w + epsilon
        output_weights[row][column] = (
            original + epsilon
        )

        logits_plus = []

        for token_id in range(
            len(output_bias)
        ):

            value = output_bias[
                token_id
            ]

            for dimension in range(
                len(decoder_vector)
            ):

                value += (
                    decoder_vector[dimension]
                    * output_weights[dimension][token_id]
                )

            logits_plus.append(value)

        loss_plus = self.loss(
            logits_plus,
            target_id
        )

        # w - epsilon
        output_weights[row][column] = (
            original - epsilon
        )

        logits_minus = []

        for token_id in range(
            len(output_bias)
        ):

            value = output_bias[
                token_id
            ]

            for dimension in range(
                len(decoder_vector)
            ):

                value += (
                    decoder_vector[dimension]
                    * output_weights[dimension][token_id]
                )

            logits_minus.append(value)

        loss_minus = self.loss(
            logits_minus,
            target_id
        )

        # Restore original value
        output_weights[row][column] = (
            original
        )

        numerical = (
            loss_plus - loss_minus
        ) / (
            2.0 * epsilon
        )

        return numerical

    def check_output_weight(
        self,
        decoder_vector,
        output_weights,
        output_bias,
        target_id,
        row,
        column
    ):

        _, output_gradient = (
            self.analytical_output_gradient(
                decoder_vector,
                output_weights,
                output_bias,
                target_id
            )
        )

        analytical = (
            decoder_vector[row]
            * output_gradient[column]
        )

        numerical = (
            self.numerical_gradient(
                decoder_vector,
                output_weights,
                output_bias,
                target_id,
                row,
                column
            )
        )

        difference = abs(
            analytical - numerical
        )

        return {
            "analytical": analytical,
            "numerical": numerical,
            "difference": difference
  }
