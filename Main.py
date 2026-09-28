from Brain.data import TextData
from Brain.tokenizer import Tokenizer
from Brain.embeddings import TokenEmbeddings
from Brain.decoder import TransformerDecoder
from Brain.loss import CrossEntropyLoss
from Brain.training import Trainer
from Brain.checkpoint import Checkpoint
from Brain.generation import TextGenerator
from Brain.gradient_check import GradientChecker
from Brain.feed_forward_gradient_check import (
    FeedForwardGradientChecker
)
from Brain.attention_gradient_check import (
    AttentionGradientChecker
)


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
# ---------------------------------------------------------
# Expanded AI and Neural Network vocabulary
# ---------------------------------------------------------

data.add("Artificial intelligence learns patterns from data")
data.add("Machine learning models learn from examples")
data.add("A neural network contains connected layers")
data.add("A neural network contains weights and biases")
data.add("Weights control the strength of connections")
data.add("Biases shift the output of a layer")
data.add("Training updates model parameters")
data.add("A dataset contains training examples")
data.add("Training data contains input examples")
data.add("A target token is the expected output")
data.add("A model predicts the next token")
data.add("A tokenizer converts text into tokens")
data.add("A vocabulary contains known tokens")
data.add("An unknown token represents unseen text")
data.add("Embeddings convert tokens into vectors")
data.add("A vector contains numerical values")
data.add("Attention compares queries and keys")
data.add("Attention produces weighted values")
data.add("Causal attention uses previous tokens")
data.add("A decoder processes token representations")
data.add("Logits represent scores for tokens")
data.add("Softmax converts logits into probabilities")
data.add("Cross entropy measures prediction error")
data.add("Backpropagation calculates gradients")
data.add("Gradients update model parameters")
data.add("An optimizer updates weights")
data.add("Learning rate controls update size")
data.add("Gradient clipping limits large updates")
data.add("A checkpoint stores model parameters")
data.add("A trained model can generate text")


# ---------------------------------------------------------
# Expanded Python vocabulary
# ---------------------------------------------------------

data.add("Python programs contain statements")
data.add("Python supports strings")
data.add("Python supports integers")
data.add("Python supports floating point numbers")
data.add("Python supports lists")
data.add("Python supports dictionaries")
data.add("Python supports loops")
data.add("Python supports conditions")
data.add("Python supports modules")
data.add("Python supports classes")
data.add("A list contains multiple values")
data.add("A dictionary stores key value pairs")
data.add("A loop repeats instructions")
data.add("A condition controls program flow")
data.add("A module contains reusable code")
data.add("An import loads a module")
data.add("A function can receive arguments")
data.add("A function can return a value")
data.add("Exceptions handle program errors")
data.add("Debugging helps find programming errors")
data.add("Testing checks program behavior")
data.add("Readable code is easier to maintain")


# ---------------------------------------------------------
# Expanded programming vocabulary
# ---------------------------------------------------------

data.add("An algorithm is a sequence of steps")
data.add("Algorithms can solve computational problems")
data.add("Data structures organize information")
data.add("Arrays store ordered values")
data.add("Stacks follow last in first out")
data.add("Queues follow first in first out")
data.add("A database stores structured information")
data.add("An API allows programs to communicate")
data.add("A server provides services to clients")
data.add("A client sends requests to a server")
data.add("Source code describes program behavior")
data.add("A compiler converts source code")
data.add("An interpreter executes program instructions")
data.add("A bug is an error in software")
data.add("A test can detect a bug")
data.add("Version control tracks code changes")
data.add("Git records changes to files")
data.add("A commit records a change")
data.add("A branch contains a line of development")
data.add("A repository stores project history")


# ---------------------------------------------------------
# Expanded Android and Kivy vocabulary
# ---------------------------------------------------------

data.add("Android applications run on mobile devices")
data.add("An APK contains an Android application")
data.add("Android applications can use permissions")
data.add("An activity represents an application screen")
data.add("A user interface contains controls")
data.add("A button can trigger an action")
data.add("A text field accepts user input")
data.add("A label displays text")
data.add("A layout arranges interface elements")
data.add("Kivy is a Python framework")
data.add("Kivy can create mobile interfaces")
data.add("Kivy applications can contain widgets")
data.add("A widget can display information")
data.add("Buildozer can package Python applications")
data.add("Buildozer uses a build configuration")
data.add("GitHub Actions can automate builds")
data.add("A workflow contains automated steps")
data.add("A build can produce an APK")
data.add("An application can store local data")
data.add("Local storage keeps information on a device")


# ---------------------------------------------------------
# Expanded mathematics vocabulary
# ---------------------------------------------------------

data.add("Addition increases a total")
data.add("Subtraction decreases a value")
data.add("Multiplication can represent repeated addition")
data.add("Division can split a quantity")
data.add("A fraction represents part of a whole")
data.add("A decimal represents a numerical value")
data.add("A percentage represents a part of one hundred")
data.add("An equation contains an equality")
data.add("A variable represents an unknown value")
data.add("An expression contains mathematical operations")
data.add("A prime number has two positive factors")
data.add("An even number is divisible by two")
data.add("An odd number is not divisible by two")
data.add("A square has equal sides")
data.add("A rectangle has opposite equal sides")
data.add("A triangle has three angles")
data.add("The perimeter measures boundary length")
data.add("Area measures a surface")
data.add("Volume measures three dimensional space")
data.add("A graph represents mathematical information")


# ---------------------------------------------------------
# Expanded physics vocabulary
# ---------------------------------------------------------

data.add("Physics studies motion and energy")
data.add("Speed measures distance per time")
data.add("Velocity includes direction")
data.add("Acceleration measures change in velocity")
data.add("Force can accelerate an object")
data.add("Mass measures the amount of matter")
data.add("Gravity acts between masses")
data.add("Friction opposes motion")
data.add("Kinetic energy is energy of motion")
data.add("Potential energy depends on position")
data.add("Energy can move between systems")
data.add("Work transfers energy through force")
data.add("Power measures the rate of energy transfer")
data.add("Light can travel through empty space")
data.add("Sound travels through a medium")
data.add("Waves transfer energy")
data.add("Temperature measures thermal state")
data.add("Electric current is flow of charge")
data.add("Voltage is a difference in electric potential")
data.add("Resistance opposes electric current")


# ---------------------------------------------------------
# Expanded chemistry vocabulary
# ---------------------------------------------------------

data.add("Chemistry studies matter and its changes")
data.add("An atom is a basic unit of matter")
data.add("Atoms contain protons and neutrons")
data.add("Electrons surround the atomic nucleus")
data.add("Protons have positive charge")
data.add("Electrons have negative charge")
data.add("Neutrons have no electric charge")
data.add("An element contains one type of atom")
data.add("A compound contains different elements")
data.add("A molecule contains bonded atoms")
data.add("Chemical reactions change substances")
data.add("Reactants participate in chemical reactions")
data.add("Products form during chemical reactions")
data.add("Water is a chemical compound")
data.add("Oxygen is a chemical element")
data.add("Hydrogen is a chemical element")
data.add("Carbon is a chemical element")
data.add("Salt can contain sodium and chlorine")
data.add("Acids and bases have different properties")
data.add("The periodic table organizes elements")


# ---------------------------------------------------------
# Expanded biology vocabulary
# ---------------------------------------------------------

data.add("Biology studies living organisms")
data.add("Cells are basic units of organisms")
data.add("Plants contain cells")
data.add("Animals contain cells")
data.add("Cells contain genetic information")
data.add("DNA stores genetic information")
data.add("Genes contain inherited information")
data.add("Plants use sunlight for photosynthesis")
data.add("Photosynthesis produces chemical energy")
data.add("Roots absorb water from soil")
data.add("Leaves perform photosynthesis")
data.add("Animals need energy to live")
data.add("The heart pumps blood")
data.add("The lungs exchange gases")
data.add("The brain processes information")
data.add("The nervous system sends signals")
data.add("Living organisms need energy")
data.add("Food provides energy and nutrients")
data.add("Ecosystems contain living organisms")
data.add("Plants and animals interact with ecosystems")


# ---------------------------------------------------------
# Expanded Layla knowledge
# ---------------------------------------------------------

data.add("Layla is a personal AI")
data.add("Layla can process user input")
data.add("Layla converts text into tokens")
data.add("Layla converts tokens into embeddings")
data.add("Layla processes embeddings with attention")
data.add("Layla uses a decoder to process text")
data.add("Layla predicts the next token")
data.add("Layla can generate text from tokens")
data.add("Layla stores learned model parameters")
data.add("Layla can load a checkpoint")
data.add("Layla can save a checkpoint")
data.add("Layla can run offline")
data.add("Layla can process local information")
data.add("Layla uses Python for its brain")
data.add("Layla uses Kivy for its interface")
data.add("Layla can be packaged as an Android application")
data.add("Layla can learn patterns from training data")
data.add("Layla improves through training")
data.add("Layla uses gradients during training")
data.add("Layla uses an optimizer during training")


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
print()
print("Attention mathematical gradient check:")

attention_checker = AttentionGradientChecker()

test_embeddings = [
    [0.10, 0.20, 0.30, 0.40],
    [0.20, 0.30, 0.40, 0.50],
    [0.30, 0.40, 0.50, 0.60]
]

test_output_gradient = [
    0.5,
    0.4,
    0.3,
    0.2
]

attention_results = (
    attention_checker.check(
        embeddings=test_embeddings,
        embedding_size=4,
        output_gradient=test_output_gradient
    )
)

attention_passed = True

for result in attention_results:

    print(
        "Position:",
        result["position"],
        "Dimension:",
        result["dimension"]
    )

    print(
        "Analytical:",
        result["analytical"]
    )

    print(
        "Numerical:",
        result["numerical"]
    )

    print(
        "Difference:",
        result["difference"]
    )

    print("-------------------------")

    if result["difference"] > 0.0001:
        attention_passed = False

if attention_passed:
    print("Attention gradient check: PASS")
else:
    print("Attention gradient check: FAIL")
# =========================================================
# FEED-FORWARD MATHEMATICAL GRADIENT CHECK
# =========================================================

ff_checker = FeedForwardGradientChecker()

ff_input = [
    0.2,
    -0.1,
    0.4,
    0.3
]

ff_w1 = [
    [0.10, -0.20, 0.30, 0.15],
    [-0.10, 0.25, -0.15, 0.20],
    [0.05, 0.10, 0.20, -0.25],
    [0.30, -0.05, 0.10, 0.15]
]

ff_b1 = [
    0.10,
    0.05,
    -0.10,
    0.20
]

ff_w2 = [
    [0.20, -0.10, 0.15, 0.05],
    [-0.15, 0.25, 0.10, -0.20],
    [0.05, 0.10, -0.25, 0.30],
    [0.10, -0.05, 0.20, 0.15]
]

ff_b2 = [
    0.10,
    -0.05,
    0.20,
    0.05
]

ff_results = ff_checker.check(
    input_vector=ff_input,
    w1=ff_w1,
    b1=ff_b1,
    w2=ff_w2,
    b2=ff_b2
)

print(
    "\nFeed-forward mathematical gradient check:"
)

ff_pass = True

for result in ff_results:

    print(
        result["parameter"],
        "Analytical:",
        result["analytical"]
    )

    print(
        result["parameter"],
        "Numerical:",
        result["numerical"]
    )

    print(
        result["parameter"],
        "Difference:",
        result["difference"]
    )

    print(
        "-------------------------"
    )

    if result["difference"] >= 0.0001:
        ff_pass = False

if ff_pass:

    print(
        "Feed-forward gradient check: PASS"
    )

else:

    print(
        "Feed-forward gradient check: FAIL"
    )

# =========================================================
# MATHEMATICAL GRADIENT CHECK
# =========================================================

gradient_checker = GradientChecker()

gradient_test_ids = tokenizer.encode_prompt(
    "Python is"
)

gradient_test_vectors = embedding.encode(
    gradient_test_ids
)

gradient_test_output = decoder.forward(
    gradient_test_vectors
)

gradient_test_vector = (
    gradient_test_output[-1]
)

gradient_target_id = tokenizer.token_to_id.get(
    "a"
)

gradient_result = (
    gradient_checker.check_output_weight(
        decoder_vector=gradient_test_vector,
        output_weights=decoder.output_weights,
        output_bias=decoder.output_bias,
        target_id=gradient_target_id,
        row=0,
        column=gradient_target_id
    )
)

print(
    "\nMathematical gradient check:"
)

print(
    "Analytical gradient:",
    gradient_result["analytical"]
)

print(
    "Numerical gradient:",
    gradient_result["numerical"]
)

print(
    "Difference:",
    gradient_result["difference"]
)

if gradient_result["difference"] < 0.0001:

    print(
        "Gradient check: PASS"
    )

else:

    print(
        "Gradient check: FAIL"
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

epochs = 50

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
