class OutputBackprop:
    def calculate_gradients(
        self,
        decoder_vector,
        output_weights,
        output_gradient
    ):
        embedding_size = len(decoder_vector)
        vocab_size = len(output_gradient)

        weight_gradients = [
            [0.0 for _ in range(vocab_size)]
            for _ in range(embedding_size)
        ]

        bias_gradients = [
            0.0 for _ in range(vocab_size)
        ]

        # dL/dW = input × dL/dlogits
        for i in range(embedding_size):
            for j in range(vocab_size):
                weight_gradients[i][j] = (
                    decoder_vector[i]
                    * output_gradient[j]
                )

        # dL/db = dL/dlogits
        for j in range(vocab_size):
            bias_gradients[j] = output_gradient[j]

        return {
            "weights": weight_gradients,
            "bias": bias_gradients
        }
