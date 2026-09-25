from Brain.data import TextData
from Brain.tokenizer import Tokenizer
from Brain.embeddings import TokenEmbeddings


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


text = "Hello Layla"

token_ids = tokenizer.encode(text)

vectors = embedding.encode(token_ids)


print("Text:", text)
print("Token IDs:", token_ids)
print("Embedding size:", len(vectors[0]))

for token_id, vector in zip(token_ids, vectors):
    print(
        "Token:",
        tokenizer.id_to_token[token_id],
        "Vector:",
        vector
    )
