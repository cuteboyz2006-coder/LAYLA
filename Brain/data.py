class TextData:

    def __init__(self):
        self.data = []

    def add(self, text):
        text = text.strip()

        if text:
            self.data.append(text)

    def get_all(self):
        return self.data

    def find_conflicts(self, tokenizer):
        prefix_targets = {}

        for text in self.data:

            token_ids = tokenizer.encode(text)

            if len(token_ids) < 2:
                continue

            input_ids = token_ids[:-1]
            target_id = token_ids[-1]

            for position in range(
                1,
                len(input_ids) + 1
            ):
                prefix = tuple(
                    input_ids[:position]
                )

                target = (
                    token_ids[position]
                    if position < len(token_ids)
                    else None
                )

                if prefix not in prefix_targets:
                    prefix_targets[prefix] = set()

                if target is not None:
                    prefix_targets[prefix].add(
                        target
                    )

        conflicts = []

        for prefix, targets in prefix_targets.items():

            if len(targets) > 1:

                conflicts.append(
                    {
                        "prefix": list(prefix),
                        "targets": list(targets)
                    }
                )

        return conflicts

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

            token_ids = token_ids[
                :sequence_length + 1
            ]

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
