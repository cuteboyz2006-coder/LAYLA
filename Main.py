from brain.data import TextData
from brain.tokenizer import Tokenizer


data = TextData()

data.add("hello Layla")
data.add("Layla is learning")
data.add("I like Python")


tokenizer = Tokenizer()

tokenizer.build_vocab(data.get_all())


text = "hello Layla"

encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded)


print("Text:", text)
print("Encoded:", encoded)
print("Decoded:", decoded)
