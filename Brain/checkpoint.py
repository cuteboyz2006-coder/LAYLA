import json
import os


class Checkpoint:

    def save(
        self,
        path,
        tokenizer,
        embedding,
        decoder
    ):
        data = {
            "tokenizer": {
                "token_to_id": tokenizer.token_to_id,
                "id_to_token": {
                    str(key): value
                    for key, value in tokenizer.id_to_token.items()
                }
            },

            "embedding": {
                "vocab_size": embedding.vocab_size,
                "embedding_size": embedding.embedding_size,
                "weights": embedding.weights
            },

            "decoder": {
                "embedding_size": decoder.embedding_size,
                "vocab_size": decoder.vocab_size,
                "hidden_size": decoder.hidden_size,

                "w1": decoder.w1,
                "b1": decoder.b1,

                "w2": decoder.w2,
                "b2": decoder.b2,

                "output_weights": decoder.output_weights,
                "output_bias": decoder.output_bias
            }
        }

        directory = os.path.dirname(path)

        if directory:
            os.makedirs(
                directory,
                exist_ok=True
            )

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file
            )

        print(
            "Checkpoint saved:",
            path
        )
