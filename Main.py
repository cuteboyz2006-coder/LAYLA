from Brain.data import TextData
from Brain.tokenizer import Tokenizer
from Brain.embeddings import TokenEmbeddings
from Brain.decoder import TransformerDecoder
from Brain.loss import CrossEntropyLoss
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
# Loss
# -------------------------

loss_function = CrossEntropyLoss()


# -------------------------
# Trainer
# -------------------------

trainer = Trainer(
    decoder=decoder,
    loss_function=loss_function,
    learning_rate=0.01
)


# -------------------------
# Training configuration
# -------------------------

epochs = 10

first_epoch_loss = None
last_epoch_loss = None


# -------------------------
# Training
# -------------------------

for epoch in range(epochs):

    total_loss = 0.0
    trained_steps = 0

    for sequence in sequences:

        input_ids = sequence["input"]
        target_ids = sequence["target"]

        if not input_ids:
            continue

        # Train every next-token position
        for position in range(
            len(input_ids)
        ):

            current_input = input_ids[
                :position + 1
            ]

            target_id = target_ids[
                position
            ]

            sequence_vectors = embedding.encode(
                current_input
            )

            loss = trainer.train_step(
                sequence_vectors,
                target_id
            )

            total_loss += loss
            trained_steps += 1

    # -------------------------
    # Average loss
    # -------------------------

    if trained_steps > 0:

        average_loss = (
            total_loss
            / trained_steps
        )

    else:

        average_loss = 0.0

    # -------------------------
    # Save first/last loss
    # -------------------------

    if first_epoch_loss is None:

        first_epoch_loss = average_loss

    last_epoch_loss = average_loss

    # -------------------------
    # Epoch output
    # -------------------------

    print(
        "Epoch:",
        epoch + 1,
        "/",
        epochs,
        "Loss:",
        average_loss
    )


# -------------------------
# Loss verification
# -------------------------

print(
    "First epoch loss:",
    first_epoch_loss
)

print(
    "Last epoch loss:",
    last_epoch_loss
)


# -------------------------
# Test input
# -------------------------

text = "Hello Layla"

token_ids = tokenizer.encode(
    text
)


# -------------------------
# Test embedding
# -------------------------

vectors = embedding.encode(
    token_ids
)


# -------------------------
# Test decoder
# -------------------------

decoder_output = decoder.forward(
    vectors
)


# -------------------------
# Test logits
# -------------------------

logits = decoder.logits(
    decoder_output
)


# -------------------------
# Test prediction
# -------------------------

next_token_id = decoder.predict_next_token(
    logits
)


# -------------------------
# Test output
# -------------------------

print(
    "Input:",
    text
)

print(
    "Token IDs:",
    token_ids
)

print(
    "Vocabulary size:",
    len(
        tokenizer.token_to_id
    )
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
