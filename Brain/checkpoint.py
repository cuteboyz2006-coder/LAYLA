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

    def load(
        self,
        path,
        tokenizer,
        embedding,
        decoder
    ):
        if not os.path.exists(path):
            raise FileNotFoundError(
                "Checkpoint not found: " + path
            )

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        # -------------------------
        # Load tokenizer
        # -------------------------

        tokenizer.token_to_id = (
            data["tokenizer"]["token_to_id"]
        )

        tokenizer.id_to_token = {
            int(key): value
            for key, value in
            data["tokenizer"]["id_to_token"].items()
        }

        # -------------------------
        # Load embeddings
        # -------------------------

        embedding.weights = (
            data["embedding"]["weights"]
        )

        # -------------------------
        # Load decoder
        # -------------------------

        decoder.w1 = (
            data["decoder"]["w1"]
        )

        decoder.b1 = (
            data["decoder"]["b1"]
        )

        decoder.w2 = (
            data["decoder"]["w2"]
        )

        decoder.b2 = (
            data["decoder"]["b2"]
        )

        decoder.output_weights = (
            data["decoder"]["output_weights"]
        )

        decoder.output_bias = (
            data["decoder"]["output_bias"]
        )

        print(
            "Checkpoint loaded:",
            path
        )
