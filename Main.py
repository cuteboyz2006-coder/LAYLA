from Brain.data import TextData
from Brain.tokenizer import Tokenizer


data = TextData()

data.add("Hello Layla")
data.add("Layla is learning")
data.add("I like Python")
data.add("Layla can learn from data")


tokenizer = Tokenizer()

tokenizer.build_vocab(data.get_all())


text = "Hello Layla"

encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded)


print("Original:", text)
print("Encoded:", encoded)
print("Decoded:", decoded)

print("Vocabulary size:", len(tokenizer.token_to_id))
print("Vocabulary:", tokenizer.token_to_id)
