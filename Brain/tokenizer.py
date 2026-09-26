class Tokenizer:
    def __init__(self):
        self.token_to_id = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3,
        }

        self.id_to_token = {
            0: "<PAD>",
            1: "<UNK>",
            2: "<BOS>",
            3: "<EOS>",
        }

    def build_vocab(self, texts):
        for text in texts:
            for token in text.lower().split():
                if token not in self.token_to_id:
                    idx = len(self.token_to_id)

                    self.token_to_id[token] = idx
                    self.id_to_token[idx] = token

    def encode(self, text):
        tokens = text.lower().split()

        ids = [
            self.token_to_id.get(token, 1)
            for token in tokens
        ]

        return [2] + ids + [3]

    def encode_prompt(self, text):
        tokens = text.lower().split()

        ids = [
            self.token_to_id.get(token, 1)
            for token in tokens
        ]

        return [2] + ids

    def decode(self, ids):
        tokens = []

        for token_id in ids:
            token = self.id_to_token.get(
                token_id,
                "<UNK>"
            )

            if token not in (
                "<PAD>",
                "<BOS>",
                "<EOS>"
            ):
                tokens.append(token)

        return " ".join(tokens)
