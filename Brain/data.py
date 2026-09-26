class TextData:

    def __init__(self):
        self.data = []

    def add(self, text):
        text = text.strip()

        if text:
            self.data.append(text)

    def get_all(self):
        return self.data

    def make_sequences(
        self,
        tokenizer,
        sequence_length=8
    ):
        sequences = []

        for text in self.data:

            token_ids = tokenizer.encode(text)

            if len(token_ids) < 2:
                continue

            # Limit the complete sequence
            token_ids = token_ids[
                :sequence_length + 1
            ]

            # --------------------------------
            # One training sequence per text
            #
            # input:
            #   BOS hello layla how
            #
            # target:
            #   hello layla how are
            # --------------------------------

            input_ids = token_ids[:-1]
            target_ids = token_ids[1:]

            if not input_ids:
                continue

            sequences.append(
                {
                    "input": input_ids,
                    "target": target_ids
                }
            )

        return sequences
