from Brain.data import TextData
from Brain.tokenizer import Tokenizer
from Brain.embeddings import TokenEmbeddings
from Brain.attention import SelfAttention


data = TextData()

data.add("Hello Layla")
data.add("Layla is learning")
data.add("I like Python")
data.add("Layla can learn from data")


tokenizer = Tokenizer()

tokenizer.build_vocab(data.get_all())


embedding = TokenEmbeddings(
    vocab_size=len(tokenizer.token_to_id),
    embedding_size=16
)


attention = SelfAttention(
    embedding_size=16
)


text = "Hello Layla"

token_ids = tokenizer.encode(text)

vectors = embedding.encode(token_ids)

attention_output = attention.forward(vectors)


print("Text:", text)
print("Token IDs:", token_ids)
print("Embedding size:", len(vectors[0]))
print("Attention output tokens:", len(attention_output))
print(
    "Attention vector size:",
    len(attention_output[0])
)

for token_id, output in zip(token_ids, attention_output):
    print(
        "Token:",
        tokenizer.id_to_token[token_id],
        "Attention:",
        output
    )
