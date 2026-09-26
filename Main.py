from Brain.data import TextData
from Brain.tokenizer import Tokenizer
from Brain.embeddings import TokenEmbeddings
from Brain.decoder import TransformerDecoder
from Brain.loss import CrossEntropyLoss
from Brain.training import Trainer
from Brain.checkpoint import Checkpoint
from Brain.generation import TextGenerator


# -------------------------
# Training data
# -------------------------
data = TextData()

# Greetings
data.add("Hello Layla")
data.add("Good morning Layla")
data.add("Good evening Layla")
data.add("Nice to meet you")
data.add("How are you today")
data.add("I am fine today")
data.add("I am happy today")
data.add("Have a nice day")

# Basic conversation
data.add("My name is Layla")
data.add("I am an AI assistant")
data.add("I can help you")
data.add("I can answer questions")
data.add("I like learning")
data.add("I enjoy helping people")

# Common words
data.add("The sun is bright")
data.add("The sky is blue")
data.add("The water is cold")
data.add("The book is useful")
data.add("The computer is fast")
data.add("The phone is useful")
data.add("The room is clean")
data.add("The game is fun")

# Learning
data.add("Python is a programming language")
data.add("Coding is useful")
data.add("Learning is important")
data.add("Practice improves skills")
data.add("Knowledge comes from learning")
data.add("Reading helps learning")
data.add("Examples help learning")

# Questions
data.add("What is your name")
data.add("What can you do")
data.add("What is Python")
data.add("What is coding")
data.add("How can you help me")
data.add("Can you learn from examples")

# Simple abilities
data.add("I can read text")
data.add("I can understand words")
data.add("I can learn from examples")
data.add("I can generate text")
data.add("I can solve simple problems")
data.add("I can remember information")

# Coding vocabulary
data.add("Python uses variables")
data.add("Python uses functions")
data.add("A function performs a task")
data.add("A variable stores information")
data.add("Code is written in files")
data.add("Programs use instructions")

# Layla
data.add("Layla is learning")
data.add("Layla can learn")
data.add("Layla can answer questions")
data.add("Layla uses a tokenizer")

conflicts = data.find_conflicts(
    tokenizer
)

print(
    "\nDataset conflicts:",
    len(conflicts)
)

for conflict in conflicts[:10]:

    prefix_text = tokenizer.decode(
        conflict["prefix"]
    )

    target_text = [
        tokenizer.id_to_token.get(
            token_id,
            "<UNK>"
        )
        for token_id in conflict["targets"]
    ]

    print(
        "Conflict:",
        prefix_text,
        "->",
        target_text
    )


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
# Target mapping diagnostic
# -------------------------

print("\nTarget mapping:")

shown = 0

for sequence in sequences:

    input_ids = sequence["input"]
    target_ids = sequence["target"]

    for position in range(
        len(input_ids)
    ):

        current_input = input_ids[
            :position + 1
        ]

        target_id = target_ids[
            position
        ]

        input_text = tokenizer.decode(
            current_input
        )

        target_token = tokenizer.id_to_token.get(
            target_id,
            "<UNK>"
        )

        print(
            input_text,
            "->",
            target_token,
            "(ID:",
            target_id,
            ")"
        )

        shown += 1

        if shown >= 20:
            break

    if shown >= 20:
        break


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
# Save embedding before training
# -------------------------

embedding_before = [
    value
    for value in embedding.weights[4]
]


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
    embedding=embedding,
    learning_rate=0.01
)


# -------------------------
# Training configuration
# -------------------------

epochs = 100

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
                target_id,
                current_input
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
# Embedding after training
# -------------------------

embedding_after = [
    value
    for value in embedding.weights[4]
]


# -------------------------
# Learning verification
# -------------------------

embedding_changed = (
    embedding_before
    != embedding_after
)

print(
    "Embedding before training:",
    embedding_before
)

print(
    "Embedding after training:",
    embedding_after
)

print(
    "Embedding changed:",
    embedding_changed
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
# Save trained checkpoint
# -------------------------

checkpoint = Checkpoint()

checkpoint.save(
    path="layla_checkpoint.json",
    tokenizer=tokenizer,
    embedding=embedding,
    decoder=decoder
)


# =========================================================
# CORRECT NEXT-TOKEN TEST
# =========================================================

test_text = "Hello Layla"

# IMPORTANT:
# encode_prompt() does NOT add EOS.
# We want to predict the token AFTER the prompt.

test_token_ids = tokenizer.encode_prompt(
    test_text
)

test_vectors = embedding.encode(
    test_token_ids
)

test_decoder_output = decoder.forward(
    test_vectors
)

test_logits = decoder.logits(
    test_decoder_output
)

test_prediction_id = (
    decoder.predict_next_token(
        test_logits
    )
)

test_prediction = (
    tokenizer.id_to_token.get(
        test_prediction_id,
        "<UNK>"
    )
)

print(
    "Test text:",
    test_text
)

print(
    "Prompt token IDs:",
    test_token_ids
)

print(
    "Prediction after training:",
    test_prediction
)


# -------------------------
# Load checkpoint verification
# -------------------------

loaded_checkpoint = Checkpoint()

loaded_checkpoint.load(
    path="layla_checkpoint.json",
    tokenizer=tokenizer,
    embedding=embedding,
    decoder=decoder
)

loaded_vectors = embedding.encode(
    test_token_ids
)

loaded_output = decoder.forward(
    loaded_vectors
)

loaded_logits = decoder.logits(
    loaded_output
)

loaded_prediction_id = (
    decoder.predict_next_token(
        loaded_logits
    )
)

loaded_prediction = (
    tokenizer.id_to_token.get(
        loaded_prediction_id,
        "<UNK>"
    )
)

print(
    "Prediction after checkpoint load:",
    loaded_prediction
)


# =========================================================
# CORRECT INPUT TEST
# =========================================================

text = "Hello Layla"

token_ids = tokenizer.encode_prompt(
    text
)

vectors = embedding.encode(
    token_ids
)

decoder_output = decoder.forward(
    vectors
)

logits = decoder.logits(
    decoder_output
)

next_token_id = decoder.predict_next_token(
    logits
)

print(
    "Input:",
    text
)

print(
    "Prompt token IDs:",
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


# -------------------------
# Fresh model checkpoint verification
# -------------------------

fresh_tokenizer = Tokenizer()

fresh_embedding = TokenEmbeddings(
    vocab_size=len(
        tokenizer.token_to_id
    ),
    embedding_size=16
)

fresh_decoder = TransformerDecoder(
    embedding_size=16,
    vocab_size=len(
        tokenizer.token_to_id
    ),
    hidden_size=32
)

fresh_checkpoint = Checkpoint()

fresh_checkpoint.load(
    path="layla_checkpoint.json",
    tokenizer=fresh_tokenizer,
    embedding=fresh_embedding,
    decoder=fresh_decoder
)

fresh_token_ids = fresh_tokenizer.encode_prompt(
    test_text
)

fresh_vectors = fresh_embedding.encode(
    fresh_token_ids
)

fresh_output = fresh_decoder.forward(
    fresh_vectors
)

fresh_logits = fresh_decoder.logits(
    fresh_output
)

fresh_prediction_id = (
    fresh_decoder.predict_next_token(
        fresh_logits
    )
)

fresh_prediction = (
    fresh_tokenizer.id_to_token.get(
        fresh_prediction_id,
        "<UNK>"
    )
)

print(
    "Prediction from fresh checkpoint model:",
    fresh_prediction
)
# =========================================================
# LEARNING DIAGNOSTIC
# =========================================================

diagnostic_tests = [
    ("I like", "python"),
    ("Layla is", "learning"),
    ("Layla can", "learn"),
]

print(
    "\nLearning diagnostic:"
)

for prompt, expected_token in diagnostic_tests:

    prompt_ids = tokenizer.encode_prompt(
        prompt
    )

    prompt_vectors = embedding.encode(
        prompt_ids
    )

    prompt_output = decoder.forward(
        prompt_vectors
    )

    prompt_logits = decoder.logits(
        prompt_output
    )

    last_logits = prompt_logits[-1]

    expected_id = tokenizer.token_to_id.get(
        expected_token
    )

    predicted_id = (
        decoder.predict_next_token(
            prompt_logits
        )
    )

    expected_score = (
        last_logits[expected_id]
        if expected_id is not None
        else None
    )

    predicted_score = (
        last_logits[predicted_id]
        if predicted_id is not None
        else None
    )

    print(
        "Prompt:",
        prompt
    )

    print(
        "Expected:",
        expected_token,
        "Score:",
        expected_score
    )

    print(
        "Predicted:",
        tokenizer.id_to_token.get(
            predicted_id,
            "<UNK>"
        ),
        "Score:",
        predicted_score
    )

    print(
        "-------------------------"
    )


# -------------------------
# Text generation test
# -------------------------

generator = TextGenerator(
    tokenizer=tokenizer,
    embedding=embedding,
    decoder=decoder
)

generated_text = generator.generate(
    text="Hello Layla",
    max_new_tokens=10
)

print(
    "Generated text:",
    generated_text
)


# =========================================================
# TOP PREDICTION DEBUG
# =========================================================

debug_token_ids = tokenizer.encode_prompt(
    "I like Python"
)

debug_vectors = embedding.encode(
    debug_token_ids
)

debug_output = decoder.forward(
    debug_vectors
)

debug_logits = decoder.logits(
    debug_output
)

last_logits = debug_logits[-1]

top_predictions = []

for token_id, score in enumerate(
    last_logits
):

    token = tokenizer.id_to_token.get(
        token_id,
        "<UNK>"
    )

    top_predictions.append(
        (score, token_id, token)
    )

top_predictions.sort(
    reverse=True
)

print(
    "Top predictions for 'I like Python':"
)

for score, token_id, token in (
    top_predictions[:10]
):

    print(
        token_id,
        token,
        score
    )


# =========================================================
# MULTIPLE GENERATION TESTS
# =========================================================

test_prompts = [
    "Hello Layla",
    "Layla is learning",
    "I like Python",
    "Layla can learn from data"
]

for prompt in test_prompts:

    result = generator.generate(
        text=prompt,
        max_new_tokens=10
    )

    print(
        "Prompt:",
        prompt
    )

    print(
        "Generated:",
        result
    )

    print(
        "-------------------------"
    )


# =========================================================
# TARGETED NEXT-TOKEN TESTS
# =========================================================

target_tests = [
    "I like",
    "Layla is",
    "Layla can"
]

for prompt in target_tests:

    prompt_ids = tokenizer.encode_prompt(
        prompt
    )

    prompt_vectors = embedding.encode(
        prompt_ids
    )

    prompt_output = decoder.forward(
        prompt_vectors
    )

    prompt_logits = decoder.logits(
        prompt_output
    )

    predicted_id = (
        decoder.predict_next_token(
            prompt_logits
        )
    )

    predicted_token = (
        tokenizer.id_to_token.get(
            predicted_id,
            "<UNK>"
        )
    )

    print(
        "Next-token test:",
        prompt,
        "->",
        predicted_token
    )
