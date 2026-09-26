from Brain.data import TextData
from Brain.tokenizer import Tokenizer
from Brain.embeddings import TokenEmbeddings
from Brain.decoder import TransformerDecoder
from Brain.loss import CrossEntropyLoss
from Brain.training import Trainer
from Brain.checkpoint import Checkpoint
from Brain.generation import TextGenerator


# =========================================================
# TRAINING DATA
# =========================================================

data = TextData()


# ---------------------------------------------------------
# Greetings
# ---------------------------------------------------------

data.add("Hello Layla")
data.add("Hello assistant")
data.add("Hello friend")
data.add("Good morning")
data.add("Good evening")
data.add("Good night")
data.add("Nice to meet you")
data.add("How are you")
data.add("How are you today")
data.add("I am fine")
data.add("I am fine today")
data.add("I am happy")
data.add("I am ready")
data.add("Have a nice day")
data.add("See you again")


# ---------------------------------------------------------
# Basic conversation
# ---------------------------------------------------------

data.add("My name is Layla")
data.add("I am an AI assistant")
data.add("I am a helpful assistant")
data.add("I like learning")
data.add("I like helping people")
data.add("I enjoy learning")
data.add("I enjoy helping")
data.add("I can help you")
data.add("I can answer questions")
data.add("I can understand text")
data.add("I can read text")
data.add("I can generate text")
data.add("I can learn from examples")
data.add("I can remember information")
data.add("I can solve problems")


# ---------------------------------------------------------
# Common vocabulary
# ---------------------------------------------------------

data.add("The sun is bright")
data.add("The sky is blue")
data.add("The moon is bright")
data.add("The stars are beautiful")
data.add("The water is cold")
data.add("The fire is hot")
data.add("The book is useful")
data.add("The computer is fast")
data.add("The phone is useful")
data.add("The room is clean")
data.add("The game is fun")
data.add("The world is large")
data.add("The earth is round")
data.add("The road is long")
data.add("The house is big")


# ---------------------------------------------------------
# Learning vocabulary
# ---------------------------------------------------------

data.add("Learning is important")
data.add("Learning is useful")
data.add("Practice improves skills")
data.add("Practice makes learning better")
data.add("Knowledge comes from learning")
data.add("Reading helps learning")
data.add("Writing improves skills")
data.add("Examples help learning")
data.add("Questions help learning")
data.add("Education is important")
data.add("Students learn new things")
data.add("Books contain information")
data.add("Teachers help students")
data.add("Study requires practice")
data.add("Memory helps learning")


# ---------------------------------------------------------
# Python vocabulary
# ---------------------------------------------------------

data.add("Python is a programming language")
data.add("Python is useful")
data.add("Python is easy to learn")
data.add("Python uses variables")
data.add("Python uses functions")
data.add("Python uses classes")
data.add("Python can process text")
data.add("Python can perform calculations")
data.add("Python programs use instructions")
data.add("Python code uses indentation")
data.add("A variable stores information")
data.add("A function performs a task")
data.add("A class defines an object")
data.add("Code is written in files")
data.add("Programs use instructions")


# ---------------------------------------------------------
# Programming vocabulary
# ---------------------------------------------------------

data.add("Coding is useful")
data.add("Coding requires practice")
data.add("Programming uses logic")
data.add("Programs contain instructions")
data.add("Software is made from code")
data.add("Algorithms solve problems")
data.add("Variables store values")
data.add("Functions perform tasks")
data.add("Objects contain data")
data.add("Classes create objects")
data.add("Debugging finds errors")
data.add("Testing finds problems")
data.add("Code can be improved")
data.add("Good code is readable")
data.add("Logic helps programming")


# ---------------------------------------------------------
# Computer vocabulary
# ---------------------------------------------------------

data.add("A computer processes information")
data.add("A computer has memory")
data.add("A computer uses a processor")
data.add("A keyboard is an input device")
data.add("A mouse is an input device")
data.add("A screen displays information")
data.add("Storage saves information")
data.add("Files contain data")
data.add("Folders contain files")
data.add("Software runs on computers")
data.add("Hardware is physical")
data.add("The processor executes instructions")
data.add("Memory stores temporary data")
data.add("A program uses memory")
data.add("Computers use operating systems")


# ---------------------------------------------------------
# AI vocabulary
# ---------------------------------------------------------

data.add("Artificial intelligence is called AI")
data.add("AI can process information")
data.add("AI can learn from data")
data.add("AI models use parameters")
data.add("Machine learning uses data")
data.add("Training changes model parameters")
data.add("A model learns patterns")
data.add("Neural networks use layers")
data.add("Neural networks use weights")
data.add("Embeddings represent tokens")
data.add("Tokens represent text")
data.add("Attention connects information")
data.add("Transformers use attention")
data.add("A decoder generates tokens")
data.add("A language model predicts tokens")


# ---------------------------------------------------------
# Layla vocabulary
# ---------------------------------------------------------

data.add("Layla is learning")
data.add("Layla is learning from data")
data.add("Layla can learn")
data.add("Layla can answer questions")
data.add("Layla can understand text")
data.add("Layla can generate text")
data.add("Layla uses a tokenizer")
data.add("Layla uses embeddings")
data.add("Layla uses attention")
data.add("Layla uses a decoder")
data.add("Layla predicts tokens")
data.add("Layla learns from examples")
data.add("Layla processes text")
data.add("Layla stores information")
data.add("Layla is an AI assistant")


# ---------------------------------------------------------
# Questions
# ---------------------------------------------------------

data.add("What is your name")
data.add("What can you do")
data.add("What is Python")
data.add("What is coding")
data.add("What is AI")
data.add("What is machine learning")
data.add("What is a computer")
data.add("What is a variable")
data.add("What is a function")
data.add("What is a token")
data.add("What is an embedding")
data.add("What is attention")
data.add("What is a transformer")
data.add("How can you help me")
data.add("How does Python work")
data.add("How does AI learn")
data.add("How does a computer work")
data.add("Can you learn from examples")
data.add("Can you understand text")
data.add("Can you answer questions")


# ---------------------------------------------------------
# Simple facts
# ---------------------------------------------------------

data.add("Water is important")
data.add("Air is important")
data.add("Plants need water")
data.add("Plants need sunlight")
data.add("Humans need food")
data.add("Humans need water")
data.add("The earth has oceans")
data.add("The earth has land")
data.add("The sun gives light")
data.add("The moon reflects sunlight")
data.add("Rain comes from clouds")
data.add("Clouds contain water")
data.add("Trees produce oxygen")
data.add("Animals need food")
data.add("Plants grow from seeds")


# ---------------------------------------------------------
# Mathematics vocabulary
# ---------------------------------------------------------

data.add("Mathematics uses numbers")
data.add("Addition combines numbers")
data.add("Subtraction finds a difference")
data.add("Multiplication combines equal groups")
data.add("Division separates numbers")
data.add("A triangle has three sides")
data.add("A square has four sides")
data.add("A rectangle has four sides")
data.add("A circle has no sides")
data.add("A right angle is ninety degrees")
data.add("Geometry studies shapes")
data.add("Algebra uses variables")
data.add("Numbers can be positive")
data.add("Numbers can be negative")
data.add("Zero is a number")


# ---------------------------------------------------------
# Science vocabulary
# ---------------------------------------------------------

data.add("Physics studies matter and energy")
data.add("Chemistry studies substances")
data.add("Biology studies living things")
data.add("Gravity attracts objects")
data.add("Force can change motion")
data.add("Energy can change form")
data.add("Light travels very fast")
data.add("Sound needs a medium")
data.add("Atoms contain smaller particles")
data.add("Molecules contain atoms")
data.add("Water contains hydrogen and oxygen")
data.add("Oxygen supports combustion")
data.add("Plants use photosynthesis")
data.add("Cells are basic units of life")
data.add("Science uses experiments")


# ---------------------------------------------------------
# Technology vocabulary
# ---------------------------------------------------------

data.add("Android is a mobile operating system")
data.add("Applications run on devices")
data.add("An APK is an Android package")
data.add("Git stores source code")
data.add("GitHub hosts repositories")
data.add("A repository contains project files")
data.add("GitHub Actions can run workflows")
data.add("A workflow runs automated tasks")
data.add("Build systems create applications")
data.add("Python can run on computers")
data.add("Kivy can create user interfaces")
data.add("Buildozer can build Android packages")
data.add("Mobile apps use interfaces")
data.add("Users interact with applications")
data.add("Software updates add features")


# =========================================================
# TOKENIZER
# =========================================================

tokenizer = Tokenizer()

tokenizer.build_vocab(
    data.get_all()
)

print(
    "\nVocabulary size:",
    len(tokenizer.token_to_id)
)


# =========================================================
# DATASET CONFLICT DIAGNOSTIC
# =========================================================

conflicts = data.find_conflicts(
    tokenizer
)

print(
    "Dataset conflicts:",
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


# =========================================================
# TRAINING SEQUENCES
# =========================================================

sequences = data.make_sequences(
    tokenizer,
    sequence_length=8
)

print(
    "Training sequences:",
    len(sequences)
)


# =========================================================
# TARGET MAPPING DIAGNOSTIC
# =========================================================

print(
    "\nTarget mapping:"
)

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

        target_token = (
            tokenizer.id_to_token.get(
                target_id,
                "<UNK>"
            )
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

        if shown >= 30:
            break

    if shown >= 30:
        break


# =========================================================
# EMBEDDINGS
# =========================================================

embedding = TokenEmbeddings(
    vocab_size=len(
        tokenizer.token_to_id
    ),
    embedding_size=16
)


# =========================================================
# SAVE EMBEDDING BEFORE TRAINING
# =========================================================

embedding_before = [
    value
    for value in embedding.weights[4]
]


# =========================================================
# TRANSFORMER DECODER
# =========================================================

decoder = TransformerDecoder(
    embedding_size=16,
    vocab_size=len(
        tokenizer.token_to_id
    ),
    hidden_size=32
)


# =========================================================
# LOSS
# =========================================================

loss_function = CrossEntropyLoss()


# =========================================================
# TRAINER
# =========================================================

trainer = Trainer(
    decoder=decoder,
    loss_function=loss_function,
    embedding=embedding,
    learning_rate=0.01
)


# =========================================================
# TRAINING CONFIGURATION
# =========================================================

epochs = 100

first_epoch_loss = None
last_epoch_loss = None


# =========================================================
# TRAINING
# =========================================================

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

            sequence_vectors = (
                embedding.encode(
                    current_input
                )
            )

            loss = trainer.train_step(
                sequence_vectors,
                target_id,
                current_input
            )

            total_loss += loss
            trained_steps += 1

    if trained_steps > 0:

        average_loss = (
            total_loss
            / trained_steps
        )

    else:

        average_loss = 0.0

    if first_epoch_loss is None:

        first_epoch_loss = average_loss

    last_epoch_loss = average_loss

    print(
        "Epoch:",
        epoch + 1,
        "/",
        epochs,
        "Loss:",
        average_loss
    )


# =========================================================
# EMBEDDING AFTER TRAINING
# =========================================================

embedding_after = [
    value
    for value in embedding.weights[4]
]


# =========================================================
# LEARNING VERIFICATION
# =========================================================

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

print(
    "First epoch loss:",
    first_epoch_loss
)

print(
    "Last epoch loss:",
    last_epoch_loss
)


# =========================================================
# SAVE CHECKPOINT
# =========================================================

checkpoint = Checkpoint()

checkpoint.save(
    path="layla_checkpoint.json",
    tokenizer=tokenizer,
    embedding=embedding,
    decoder=decoder
)


# =========================================================
# NEXT TOKEN TEST
# =========================================================

test_text = "Hello Layla"

test_token_ids = (
    tokenizer.encode_prompt(
        test_text
    )
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


# =========================================================
# CHECKPOINT LOAD VERIFICATION
# =========================================================

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
# INPUT TEST
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

next_token_id = (
    decoder.predict_next_token(
        logits
    )
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


# =========================================================
# FRESH MODEL CHECKPOINT VERIFICATION
# =========================================================

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

fresh_token_ids = (
    fresh_tokenizer.encode_prompt(
        test_text
    )
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
    ("I like", "learning"),
    ("Layla is", "learning"),
    ("Layla can", "learn"),
    ("Python is", "a"),
    ("The sun is", "bright"),
]

print(
    "\nLearning diagnostic:"
)

for prompt, expected_token in diagnostic_tests:

    prompt_ids = (
        tokenizer.encode_prompt(
            prompt
        )
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

    if not prompt_logits:
        continue

    last_logits = prompt_logits[-1]

    expected_id = (
        tokenizer.token_to_id.get(
            expected_token
        )
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


# =========================================================
# TEXT GENERATION TEST
# =========================================================

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

debug_token_ids = (
    tokenizer.encode_prompt(
        "I like Python"
    )
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

if debug_logits:

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
            (
                score,
                token_id,
                token
            )
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
    "Python is a",
    "The sun is",
    "Layla can learn"
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
# TARGETED NEXT TOKEN TESTS
# =========================================================

target_tests = [
    "I like",
    "Layla is",
    "Layla can",
    "Python is",
    "The sun is"
]

for prompt in target_tests:

    prompt_ids = (
        tokenizer.encode_prompt(
            prompt
        )
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
