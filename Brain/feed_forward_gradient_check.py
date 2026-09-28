import math


class FeedForwardGradientChecker:

    def forward(
        self,
        input_vector,
        w1,
        b1,
        w2,
        b2
    ):
        hidden = []

        for j in range(len(b1)):

            value = b1[j]

            for i in range(len(input_vector)):
                value += (
                    input_vector[i]
                    * w1[i][j]
                )

            hidden.append(
                max(0.0, value)
            )

        output = []

        for j in range(len(b2)):

            value = b2[j]

            for i in range(len(hidden)):
                value += (
                    hidden[i]
                    * w2[i][j]
                )

            output.append(value)

        return hidden, output

    def numerical_gradient(
        self,
        input_vector,
        w1,
        b1,
        w2,
        b2,
        parameter_type,
        row=0,
        column=0,
        epsilon=0.000001
    ):

        def calculate_loss():

            _, output = self.forward(
                input_vector,
                w1,
                b1,
                w2,
                b2
            )

            # Simple scalar function:
            # loss = sum(output)
            return sum(output)

        if parameter_type == "w1":

            original = w1[row][column]

            w1[row][column] = (
                original + epsilon
            )

            loss_plus = calculate_loss()

            w1[row][column] = (
                original - epsilon
            )

            loss_minus = calculate_loss()

            w1[row][column] = original

        elif parameter_type == "w2":

            original = w2[row][column]

            w2[row][column] = (
                original + epsilon
            )

            loss_plus = calculate_loss()

            w2[row][column] = (
                original - epsilon
            )

            loss_minus = calculate_loss()

            w2[row][column] = original

        elif parameter_type == "b1":

            original = b1[row]

            b1[row] = (
                original + epsilon
            )

            loss_plus = calculate_loss()

            b1[row] = (
                original - epsilon
            )

            loss_minus = calculate_loss()

            b1[row] = original

        elif parameter_type == "b2":

            original = b2[row]

            b2[row] = (
                original + epsilon
            )

            loss_plus = calculate_loss()

            b2[row] = (
                original - epsilon
            )

            loss_minus = calculate_loss()

            b2[row] = original

        else:

            raise ValueError(
                "Unknown parameter type"
            )

        return (
            loss_plus - loss_minus
        ) / (
            2.0 * epsilon
        )

    def analytical_gradients(
        self,
        input_vector,
        w1,
        b1,
        w2,
        b2
    ):

        hidden, output = self.forward(
            input_vector,
            w1,
            b1,
            w2,
            b2
        )

        # Because:
        #
        # loss = sum(output)
        #
        # dLoss/dOutput = 1
        output_gradient = [
            1.0
            for _ in output
        ]

        # W2 gradient
        w2_gradient = [
            [
                0.0
                for _ in range(len(b2))
            ]
            for _ in range(len(b1))
        ]

        for i in range(len(hidden)):

            for j in range(len(b2)):

                w2_gradient[i][j] = (
                    hidden[i]
                    * output_gradient[j]
                )

        # b2 gradient
        b2_gradient = [
            1.0
            for _ in range(len(b2))
        ]

        # Hidden gradient
        hidden_gradient = [
            0.0
            for _ in range(len(hidden))
        ]

        for i in range(len(hidden)):

            value = 0.0

            for j in range(len(b2)):

                value += (
                    w2[i][j]
                    * output_gradient[j]
                )

            hidden_gradient[i] = value

        # ReLU gradient
        relu_gradient = []

        for value in hidden:

            if value > 0.0:
                relu_gradient.append(1.0)
            else:
                relu_gradient.append(0.0)

        for i in range(len(hidden)):

            hidden_gradient[i] *= (
                relu_gradient[i]
            )

        # W1 gradient
        w1_gradient = [
            [
                0.0
                for _ in range(len(b1))
            ]
            for _ in range(len(input_vector))
        ]

        for i in range(len(input_vector)):

            for j in range(len(b1)):

                w1_gradient[i][j] = (
                    input_vector[i]
                    * hidden_gradient[j]
                )

        # b1 gradient
        b1_gradient = [
            value
            for value in hidden_gradient
        ]

        return {
            "w1": w1_gradient,
            "b1": b1_gradient,
            "w2": w2_gradient,
            "b2": b2_gradient
        }

    def check(
        self,
        input_vector,
        w1,
        b1,
        w2,
        b2
    ):

        analytical = self.analytical_gradients(
            input_vector,
            w1,
            b1,
            w2,
            b2
        )

        tests = [
            ("w1", 0, 0),
            ("w2", 0, 0),
            ("b1", 0, 0),
            ("b2", 0, 0)
        ]

        results = []

        for parameter_type, row, column in tests:

            numerical = self.numerical_gradient(
                input_vector,
                w1,
                b1,
                w2,
                b2,
                parameter_type,
                row,
                column
            )

            if parameter_type == "w1":

                analytical_value = (
                    analytical["w1"][row][column]
                )

            elif parameter_type == "w2":

                analytical_value = (
                    analytical["w2"][row][column]
                )

            elif parameter_type == "b1":

                analytical_value = (
                    analytical["b1"][row]
                )

            else:

                analytical_value = (
                    analytical["b2"][row]
                )

            difference = abs(
                analytical_value
                - numerical
            )

            results.append(
                {
                    "parameter": (
                        parameter_type
                    ),
                    "analytical": (
                        analytical_value
                    ),
                    "numerical": numerical,
                    "difference": difference
                }
            )

        return results
