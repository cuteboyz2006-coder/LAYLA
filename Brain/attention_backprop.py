class AttentionBackprop:
    def calculate_input_gradient(
        self,
        embeddings,
        output_gradient,
        embedding_size
    ):
        if not embeddings:
            return []

        input_gradients = [
            [0.0 for _ in range(embedding_size)]
            for _ in range(len(embeddings))
        ]

        # --------------------------------
        # Direct value-path gradient
        # --------------------------------

        last_position = len(embeddings) - 1

        gradient = output_gradient

        visible_count = last_position + 1

        for position in range(visible_count):
            weight = 1.0 / visible_count

            for dimension in range(
                embedding_size
            ):
                input_gradients[position][dimension] += (
                    weight * gradient[dimension]
                )

        return input_gradients
