from Brain.data import TextData
from Brain.tokenizer import Tokenizer


data = TextData()

data.add("Hello Layla")
data.add("Layla is learning")
data.add("I like Python")
data.add("Layla can learn from data")


tokenizer = Tokenizer()

tokenizer.build_vocab(data.get_all())


sequences = data.make_sequences(
    tokenizer,
    sequence_length=8
)


print("Training examples:", len(sequences))

for example in sequences[:10]:
    print(
        "Input:",
        example["input"],
        "Target:",
        example["target"]
    )
