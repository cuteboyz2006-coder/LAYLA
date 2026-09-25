from Brain.data import TextData
from Brain.tokenizer import Tokenizer
from Brain.embeddings import TokenEmbeddings
from Brain.decoder import TransformerDecoder


# -------------------------
# Training data
# -------------------------

data = TextData()

data.add("Hello Layla")
data.add("Layla is learning")
data.add("I like Python")
data.add("Layla can learn from data")


# -------------------------
# Tokenizer
# -------------------------

tokenizer = Tokenizer()

tokenizer.build_vocab(
    data.get_all()
)


# -------------------------
# Embeddings
# -------------------------

embedding = TokenEmbeddings(
    vocab_size=len(
        tokenizer.token_to_id
    ),
    embedding_size=16
)


# -------------------------
# Transformer Decoder
# -------------------------

decoder = TransformerDecoder(
    embedding_size=16,
    vocab_size=len(
        tokenizer.token_to_id
    ),
    hidden_size=32
)


# -------------------------
# Input
# -------------------------

text = "Hello Layla"

token_ids = tokenizer.encode(
    text
)


# -------------------------
# Embedding
# -------------------------

vectors = embedding.encode(
    token_ids
)


# -------------------------
# Decoder forward pass
# -------------------------

decoder_output = decoder.forward(
    vectors
)


# -------------------------
# Vocabulary logits
# -------------------------

logits = decoder.logits(
    decoder_output
)


# -------------------------
# Next-token prediction
# -------------------------

next_token_id = decoder.predict_next_token(
    logits
)


print("Input:", text)

print(
    "Token IDs:",
    token_ids
)

print(
    "Embedding size:",
    len(vectors[0])
)

print(
    "Decoder tokens:",
    len(decoder_output)
)

print(
    "Vocabulary size:",
    len(tokenizer.token_to_id)
)

print(
    "Logits size:",
    len(logits[-1])
)

print(
    "Predicted token ID:",
    next_token_id
)

print(
    "Predicted token:",
    tokenizer.id_to_token.get(
        next_token_id,
        "<UNK>"
    )
)
