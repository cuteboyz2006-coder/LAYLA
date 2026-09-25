class TextData:
    def __init__(self):
        self.data = []

    def add(self, text):
        text = text.strip()

        if text:
            self.data.append(text)

    def get_all(self):
        return self.data

    def make_sequences(self, tokenizer, sequence_length=8):
        sequences = []

        for text in self.data:
            token_ids = tokenizer.encode(text)

            if len(token_ids) < 2:
                continue

            for i in range(len(token_ids) - 1):
                input_ids = token_ids[:i + 1]
                target_ids = token_ids[1:i + 2]

                if len(input_ids) > sequence_length:
                    break

                sequences.append(
                    {
                        "input": input_ids,
                        "target": target_ids,
                    }
                )

        return sequences
