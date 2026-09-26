class FeedForwardBackprop:
    def relu_gradient(self, vector):
        gradients = []

        for value in vector:
            if value > 0:
                gradients.append(1.0)
            else:
                gradients.append(0.0)

        return gradients

    def calculate(
        self,
        input_vector,
        hidden_vector,
        output_gradient,
        w1,
        w2
    ):
        input_size = len(input_vector)
        hidden_size = len(hidden_vector)
        output_size = len(output_gradient)

        # -------------------------
        # Gradient through W2
        # -------------------------

        w2_gradient = [
            [0.0 for _ in range(output_size)]
            for _ in range(hidden_size)
        ]

        b2_gradient = [
            0.0
            for _ in range(output_size)
        ]

        for i in range(hidden_size):
            for j in range(output_size):
                w2_gradient[i][j] = (
                    hidden_vector[i]
                    * output_gradient[j]
                )

        for j in range(output_size):
            b2_gradient[j] = output_gradient[j]

        # -------------------------
        # Gradient to hidden layer
        # -------------------------

        hidden_gradient = [
            0.0
            for _ in range(hidden_size)
        ]

        for i in range(hidden_size):
            value = 0.0

            for j in range(output_size):
                value += (
                    w2[i][j]
                    * output_gradient[j]
                )

            hidden_gradient[i] = value

        # -------------------------
        # ReLU backward
        # -------------------------

        relu_gradient = self.relu_gradient(
            hidden_vector
        )

        for i in range(hidden_size):
            hidden_gradient[i] *= (
                relu_gradient[i]
            )

        # -------------------------
        # Gradient through W1
        # -------------------------

        w1_gradient = [
            [0.0 for _ in range(hidden_size)]
            for _ in range(input_size)
        ]

        b1_gradient = [
            0.0
            for _ in range(hidden_size)
        ]

        for i in range(input_size):
            for j in range(hidden_size):
                w1_gradient[i][j] = (
                    input_vector[i]
                    * hidden_gradient[j]
                )

        for j in range(hidden_size):
            b1_gradient[j] = hidden_gradient[j]

        # -------------------------
        # Gradient to input
        # -------------------------

        input_gradient = [
            0.0
            for _ in range(input_size)
        ]

        for i in range(input_size):
            value = 0.0

            for j in range(hidden_size):
                value += (
                    w1[i][j]
                    * hidden_gradient[j]
                )

            input_gradient[i] = value

        return {
            "w1": w1_gradient,
            "b1": b1_gradient,
            "w2": w2_gradient,
            "b2": b2_gradient,
            "input": input_gradient
        }
