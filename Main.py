from Brain.data import TextData
from Brain.tokenizer import Tokenizer
from Brain.embeddings import TokenEmbeddings
from Brain.decoder import TransformerDecoder
from Brain.loss import CrossEntropyLoss
from Brain.backprop import OutputBackprop
from Brain.optimizer import SGD
from Brain.training import Trainer


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
# Training sequences
# -------------------------

sequences = data.make_sequences(
    tokenizer,
    sequence_length=8
)

print(
    "Training sequences:",
    len(sequences)
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
# Training loop test
# -------------------------

trainer = Trainer(
    decoder=decoder,
    loss_function=loss_function,
    learning_rate=0.01
)

# -------------------------
# Train on all sequences
# -------------------------

total_loss = 0.0
trained_sequences = 0

for sequence in sequences:
    input_ids = sequence["input"]
    target_ids = sequence["target"]

    if not input_ids:
        continue

    sequence_vectors = embedding.encode(
        input_ids
    )

    target_id = target_ids[-1]

    loss = trainer.train_step(
        sequence_vectors,
        target_id
    )

    total_loss += loss
    trained_sequences += 1


# -------------------------
# Average training loss
# -------------------------

if trained_sequences > 0:
    average_loss = (
        total_loss
        / trained_sequences
    )
else:
    average_loss = 0.0


print(
    "Trained sequences:",
    trained_sequences
)

print(
    "Average training loss:",
    average_loss
)
)

print(
    "Training step loss:",
    training_loss
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
