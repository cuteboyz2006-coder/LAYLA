class TextGenerator:
    def __init__(
        self,
        tokenizer,
        embedding,
        decoder
    ):
        self.tokenizer = tokenizer
        self.embedding = embedding
        self.decoder = decoder

    def generate(
        self,
        text,
        max_new_tokens=10
    ):
        token_ids = self.tokenizer.encode(text)

        generated_ids = []

        for _ in range(max_new_tokens):

            vectors = self.embedding.encode(
                token_ids
            )

            decoder_output = self.decoder.forward(
                vectors
            )

            logits = self.decoder.logits(
                decoder_output
            )

            next_token_id = (
                self.decoder.predict_next_token(
                    logits
                )
            )

            if next_token_id is None:
                break

            if next_token_id == 3:
                break

            token_ids.append(
                next_token_id
            )

            generated_ids.append(
                next_token_id
            )

        return self.tokenizer.decode(
            generated_ids
        )
