class Tokenizer:
    def __init__(self):
        self.token_to_id = {
            "<PAD>": 0,
            "<UNK>": 1,
        }

        self.id_to_token = {
            0: "<PAD>",
            1: "<UNK>",
        }

    def build_vocab(self, texts):
        for text in texts:
            for token in text.lower().split():
                if token not in self.token_to_id:
                    idx = len(self.token_to_id)

                    self.token_to_id[token] = idx
                    self.id_to_token[idx] = token

    def encode(self, text):
        return [
            self.token_to_id.get(token, 1)
            for token in text.lower().split()
        ]

    def decode(self, ids):
        return " ".join(
            self.id_to_token.get(i, "<UNK>")
            for i in ids
        )
