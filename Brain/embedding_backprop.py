class EmbeddingBackprop:
    def update_embedding_gradient(
        self,
        embedding_weights,
        token_ids,
        input_gradients
    ):
        gradients = [
            [0.0 for _ in vector]
            for vector in embedding_weights
        ]

        for position, token_id in enumerate(token_ids):

            if token_id < 0:
                continue

            if token_id >= len(
                embedding_weights
            ):
                continue

            if position >= len(
                input_gradients
            ):
                continue

            for dimension in range(
                len(embedding_weights[token_id])
            ):
                gradients[token_id][dimension] += (
                    input_gradients[position][dimension]
                )

        return gradients
