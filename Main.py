from Brain.data import TextData
from Brain.tokenizer import Tokenizer
from Brain.embeddings import TokenEmbeddings
from Brain.decoder import TransformerDecoder
from Brain.loss import CrossEntropyLoss
from Brain.backprop import OutputBackprop
from Brain.optimizer import SGD


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
# Decoder
# -------------------------

decoder = TransformerDecoder(
    embedding_size=16,
    vocab_size=len(
        tokenizer.token_to_id
    ),
    hidden_size=32
)


# -------------------------
# Loss
# -------------------------

loss_function = CrossEntropyLoss()


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
# Decoder
# -------------------------

decoder_output = decoder.forward(
    vectors
)


# -------------------------
# Logits
# -------------------------

logits = decoder.logits(
    decoder_output
)


# -------------------------
# Target
# -------------------------

target_id = token_ids[-1]


# -------------------------
# Loss
# -------------------------

loss = loss_function.loss(
    logits[-1],
    target_id
)


# -------------------------
# Gradient
# -------------------------

gradient = loss_function.gradient(
    logits[-1],
    target_id
)
# -------------------------
# Output backpropagation
# -------------------------

backprop = OutputBackprop()

gradients = backprop.calculate_gradients(
    decoder_output[-1],
    decoder.output_weights,
    gradient
)

print(
    "Output weight gradient rows:",
    len(gradients["weights"])
)

print(
    "Output weight gradient columns:",
    len(gradients["weights"][0])
)

print(
    "Bias gradient size:",
    len(gradients["bias"])
)
loss_before = loss
# -------------------------
# Optimizer
# -------------------------

optimizer = SGD(
    learning_rate=0.01
)

optimizer.update_matrix(
    decoder.output_weights,
    gradients["weights"]
)

optimizer.update_vector(
    decoder.output_bias,
    gradients["bias"]
)

print(
    "Output weights updated:",
    True
)

print(
    "Output bias updated:",
    True
)
# -------------------------
# Check learning
# -------------------------

new_decoder_output = decoder.forward(
    vectors
)

new_logits = decoder.logits(
    new_decoder_output
)

loss_after = loss_function.loss(
    new_logits[-1],
    target_id
)

print(
    "Loss before:",
    loss_before
)

print(
    "Loss after:",
    loss_after
)


# -------------------------
# Test output
# -------------------------

print("Input:", text)

print(
    "Target token ID:",
    target_id
)

print(
    "Target token:",
    tokenizer.id_to_token.get(
        target_id,
        "<UNK>"
    )
)

print(
    "Loss:",
    loss
)

print(
    "Gradient size:",
    len(gradient)
)

print(
    "Gradient sample:",
    gradient[:5]
)
