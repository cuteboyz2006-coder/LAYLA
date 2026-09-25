class DecoderInputBackprop:
    def calculate_gradient(
        self,
        output_weights,
        output_gradient
    ):
        embedding_size = len(output_weights)
        vocab_size = len(output_gradient)

        decoder_gradient = [
            0.0
            for _ in range(embedding_size)
        ]

        for i in range(embedding_size):

            value = 0.0

            for j in range(vocab_size):

                value += (
                    output_weights[i][j]
                    * output_gradient[j]
                )

            decoder_gradient[i] = value

        return decoder_gradient
