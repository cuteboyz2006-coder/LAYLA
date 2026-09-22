import os
import re
import json
import ast
import operator
import math
import random
import difflib

from kivy.app import App
from kivy.core.window import Window
from kivy.metrics import dp
from kivy.graphics import Color, RoundedRectangle
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.uix.popup import Popup

try:
    from plyer import filechooser
except Exception:
    filechooser = None


# ============================================================
# SAFE MATH
# ============================================================

class SafeMath:

    OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    @classmethod
    def evaluate(cls, expression):

        expression = expression.replace("×", "*")
        expression = expression.replace("÷", "/")
        expression = expression.replace("^", "**")

        tree = ast.parse(expression, mode="eval")

        return cls._eval(tree.body)

    @classmethod
    def _eval(cls, node):

        if isinstance(node, ast.Constant):

            if isinstance(node.value, (int, float)):
                return node.value

            raise ValueError("Invalid number")

        if isinstance(node, ast.Num):
            return node.n

        if isinstance(node, ast.BinOp):

            if type(node.op) not in cls.OPS:
                raise ValueError("Operator not allowed")

            left = cls._eval(node.left)
            right = cls._eval(node.right)

            if isinstance(node.op, ast.Pow):

                if abs(right) > 100:
                    raise ValueError("Power too large")

            return cls.OPS[type(node.op)](left, right)

        if isinstance(node, ast.UnaryOp):

            if type(node.op) not in cls.OPS:
                raise ValueError("Operator not allowed")

            return cls.OPS[type(node.op)](
                cls._eval(node.operand)
            )

        raise ValueError("Invalid expression")


# ============================================================
# TEXT NORMALIZER
# ============================================================

class TextNormalizer:

    @staticmethod
    def normalize(text):

        text = text.strip().lower()

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        replacements = {

            "hw r u": "how are you",
            "how r u": "how are you",
            "how r you": "how are you",
            "how r u?": "how are you",
            "hru": "how are you",

            "kese ho": "kaise ho",
            "kaisa hai": "kaise ho",
            "kaisi ho": "kaise ho",
            "kaise h": "kaise ho",
            "kaise ho": "kaise ho",

            "thik ho": "theek ho",
            "thk ho": "theek ho",

            "okey": "okay",
            "okk": "okay",

            "helo": "hello",
            "hlo": "hello",
            "hlw": "hello",

            "hii": "hi",
            "heyy": "hey",
        }

        return replacements.get(
            text,
            text
        )


# ============================================================
# SPELL CORRECTOR
# ============================================================

class SpellCorrector:

    WORDS = [

        "hello",
        "hi",
        "hey",
        "kaise",
        "ho",
        "theek",
        "kya",
        "hai",
        "haan",
        "nahi",
        "mera",
        "meri",
        "naam",
        "tumhara",
        "time",
        "date",
        "help",
        "math",
        "study",
        "homework",
        "physics",
        "chemistry",
        "biology",
        "coding",
        "python",
        "java",
        "javascript",
        "story",
        "script",
        "caption",
        "game",
        "gaming",
        "thank",
        "thanks",
        "good",
        "morning",
        "night",
    ]

    @classmethod
    def correct(cls, text):

        parts = text.split()

        result = []

        for word in parts:

            clean = re.sub(
                r"[^a-zA-Z]",
                "",
                word.lower()
            )

            if len(clean) < 4:

                result.append(word)

                continue

            match = difflib.get_close_matches(
                clean,
                cls.WORDS,
                n=1,
                cutoff=0.82
            )

            if match and match[0] != clean:

                word = re.sub(
                    re.escape(clean),
                    match[0],
                    word,
                    flags=re.IGNORECASE
                )

            result.append(word)

        return " ".join(result)


# ============================================================
# AUTOMATIC MEMORY
# ============================================================

class OfflineBrain:

    def __init__(self):

        app = App.get_running_app()

        self.file = os.path.join(
            app.user_data_dir,
            "knowledge.json"
        )

        self.data = {}

        self.load()

    def load(self):

        try:

            if os.path.exists(self.file):

                with open(
                    self.file,
                    "r",
                    encoding="utf-8"
                ) as f:

                    self.data = json.load(f)

        except Exception:

            self.data = {}

    def save(self):

        try:

            folder = os.path.dirname(
                self.file
            )

            os.makedirs(
                folder,
                exist_ok=True
            )

            with open(
                self.file,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    self.data,
                    f,
                    ensure_ascii=False,
                    indent=2
                )

        except Exception:

            pass

    def learn(self, text):

        original = text.strip()

        lower = original.lower()

        patterns = [

            (
                r"\bmy name is\s+(.+)",
                "name"
            ),

            (
                r"\bmera naam\s+(.+)",
                "name"
            ),

            (
                r"\bmy favourite game is\s+(.+)",
                "favourite_game"
            ),

            (
                r"\bmy favorite game is\s+(.+)",
                "favourite_game"
            ),

            (
                r"\bmujhe\s+(.+?)\s+pasand hai",
                "likes"
            ),

            (
                r"\bi like\s+(.+)",
                "likes"
            ),

            (
                r"\bi love\s+(.+)",
                "likes"
            ),

            (
                r"\bi study in\s+(.+)",
                "school"
            ),
        ]

        for pattern, key in patterns:

            match = re.search(
                pattern,
                lower
            )

            if match:

                value = match.group(1).strip()

                if value:

                    self.data[key] = value

                    self.save()

                    return True

        return False

    def answer(self, text):

        lower = text.lower().strip()

        if "what is my name" in lower:

            if "name" in self.data:

                return (
                    "Tumhara naam "
                    + str(self.data["name"])
                    + " hai."
                )

            return (
                "Tumne abhi tak apna naam nahi bataya."
            )

        if "mera naam kya hai" in lower:

            if "name" in self.data:

                return (
                    "Tumhara naam "
                    + str(self.data["name"])
                    + " hai."
                )

            return (
                "Tumne abhi tak apna naam nahi bataya."
            )

        if (
            "my favourite game" in lower
            or
            "my favorite game" in lower
        ):

            if "favourite_game" in self.data:

                return (
                    "Tumhara favourite game "
                    + str(self.data["favourite_game"])
                    + " hai."
                )

        if (
            "what do i like" in lower
            or
            "mujhe kya pasand hai" in lower
        ):

            if "likes" in self.data:

                return (
                    "Tumne bataya tha ki tumhe "
                    + str(self.data["likes"])
                    + " pasand hai."
                )

        return None


# ============================================================
# CONVERSATION
# ============================================================

class ConversationEngine:

    @staticmethod
    def solve(text):

        t = TextNormalizer.normalize(text)

        greetings = {
            "hi",
            "hello",
            "hey",
            "hii",
            "namaste",
            "namaskar"
        }

        if t in greetings:

            return (
                "Hello! Main Layla hoon. "
                "Batao, main tumhari kis cheez mein help karun?"
            )

        if t in (
            "kaise ho",
            "how are you",
            "theek ho",
            "tum kaise ho"
        ):

            return (
                "Main bilkul ready hoon. "
                "Tum batao, aaj kya karna hai?"
            )

        if t in (
            "good morning",
            "gm"
        ):

            return (
                "Good morning. "
                "Aaj ka din productive banate hain."
            )

        if t in (
            "good night",
            "gn"
        ):

            return (
                "Good night. Kal phir milte hain."
            )

        if t in (
            "thank you",
            "thanks",
            "thank"
        ):

            return "You're welcome."

        if (
            "who are you" in t
            or
            "tum kaun ho" in t
        ):

            return (
                "Main Layla hoon, tumhari personal AI assistant. "
                "Main chat, maths, study, coding aur creative tasks "
                "mein help kar sakti hoon."
            )

        if (
            "your name" in t
            or
            "tumhara naam" in t
        ):

            return "Mera naam Layla hai."

        if (
            "what can you do" in t
            or
            "kya kar sakti ho" in t
        ):

            return (
                "Main conversation, maths, study, homework, coding, "
                "creative writing, reasoning, general knowledge aur "
                "basic file/photo/video selection mein help kar sakti hoon."
            )

        return None


# ============================================================
# MATH
# ============================================================

class MathEngine:

    @staticmethod
    def solve(text):

        original = text.lower().strip()

        expr = original

        for word in [
            "what is",
            "calculate",
            "solve",
            "equals",
            "kitna",
            "batao"
        ]:

            expr = expr.replace(
                word,
                ""
            )

        expr = expr.strip()

        if not re.search(
            r"\d",
            expr
        ):

            return None

        if not re.search(
            r"[\+\-\*\/×÷\^%]",
            expr
        ):

            return None

        if not re.fullmatch(
            r"[\d\s\+\-\*\/×÷\^\%\(\)\.]+",
            expr
        ):

            return None

        try:

            result = SafeMath.evaluate(
                expr
            )

            if isinstance(
                result,
                float
            ):

                if result.is_integer():

                    result = int(result)

                else:

                    result = round(
                        result,
                        10
                    )

            return (
                "Answer: "
                + str(result)
            )

        except Exception:

            return None


# ============================================================
# ADVANCED MATH
# ============================================================

class AdvancedMathEngine:

    @staticmethod
    def solve(text):

        t = text.lower()

        number = re.search(
            r"(-?\d+(?:\.\d+)?)",
            t
        )

        try:

            if number:

                x = float(
                    number.group(1)
                )

                if (
                    "square root" in t
                    or
                    "sqrt" in t
                ):

                    return (
                        "Answer: "
                        + str(math.sqrt(x))
                    )

                if "sin" in t:

                    return (
                        "Answer: "
                        + str(
                            round(
                                math.sin(
                                    math.radians(x)
                                ),
                                10
                            )
                        )
                    )

                if "cos" in t:

                    return (
                        "Answer: "
                        + str(
                            round(
                                math.cos(
                                    math.radians(x)
                                ),
                                10
                            )
                        )
                    )

                if "tan" in t:

                    return (
                        "Answer: "
                        + str(
                            round(
                                math.tan(
                                    math.radians(x)
                                ),
                                10
                            )
                        )
                    )

                if "log" in t:

                    return (
                        "Answer: "
                        + str(
                            round(
                                math.log10(x),
                                10
                            )
                        )
                    )

                if "ln" in t:

                    return (
                        "Answer: "
                        + str(
                            round(
                                math.log(x),
                                10
                            )
                        )
                    )

                if "factorial" in t:

                    return (
                        "Answer: "
                        + str(
                            math.factorial(
                                int(x)
                            )
                        )
                    )

        except Exception:

            return (
                "I could not calculate that."
            )

        return None


# ============================================================
# GEOMETRY
# ============================================================

class GeometryEngine:

    @staticmethod
    def solve(text):

        t = text.lower()

        nums = re.findall(
            r"-?\d+(?:\.\d+)?",
            t
        )

        try:

            if (
                "area of circle" in t
                and nums
            ):

                r = float(nums[0])

                return (
                    "Area = "
                    + str(
                        round(
                            math.pi * r * r,
                            4
                        )
                    )
                )

            if (
                "circumference" in t
                and nums
            ):

                r = float(nums[0])

                return (
                    "Circumference = "
                    + str(
                        round(
                            2 * math.pi * r,
                            4
                        )
                    )
                )

            if (
                "area of rectangle" in t
                and len(nums) >= 2
            ):

                l = float(nums[0])
                w = float(nums[1])

                return (
                    "Area = "
                    + str(l * w)
                )

            if (
                "area of triangle" in t
                and len(nums) >= 2
            ):

                b = float(nums[0])
                h = float(nums[1])

                return (
                    "Area = "
                    + str(
                        0.5 * b * h
                    )
                )

            if (
                "pythagoras" in t
                and len(nums) >= 2
            ):

                a = float(nums[0])
                b = float(nums[1])

                c = math.sqrt(
                    a * a + b * b
                )

                return (
                    "Hypotenuse = "
                    + str(
                        round(c, 6)
                    )
                )

        except Exception:

            return None

        return None


# ============================================================
# UNIT CONVERTER
# ============================================================

class UnitEngine:

    FACTORS = {

        ("km", "m"): 1000,
        ("m", "km"): 0.001,

        ("m", "cm"): 100,
        ("cm", "m"): 0.01,

        ("kg", "g"): 1000,
        ("g", "kg"): 0.001,

        ("hour", "minute"): 60,
        ("minute", "second"): 60,

        ("second", "minute"): 1 / 60,
        ("minute", "hour"): 1 / 60,

        ("l", "ml"): 1000,
        ("ml", "l"): 0.001,
    }

    @classmethod
    def solve(cls, text):

        t = text.lower()

        t = t.replace(
            " to ",
            " "
        )

        pattern = (
            r"(-?\d+(?:\.\d+)?)"
            r"\s*([a-z]+)"
            r"\s+([a-z]+)"
        )

        match = re.search(
            pattern,
            t
        )

        if not match:

            return None

        value = float(
            match.group(1)
        )

        source = match.group(2)
        target = match.group(3)

        key = (
            source,
            target
        )

        if key in cls.FACTORS:

            result = (
                value
                * cls.FACTORS[key]
            )

            if result.is_integer():

                result = int(result)

            return (
                str(value)
                + " "
                + source
                + " = "
                + str(result)
                + " "
                + target
            )

        return None


# ============================================================
# PHYSICS
# ============================================================

class PhysicsEngine:

    @staticmethod
    def solve(text):

        t = text.lower()

        nums = re.findall(
            r"-?\d+(?:\.\d+)?",
            t
        )

        try:

            if (
                "speed" in t
                and len(nums) >= 2
            ):

                distance = float(nums[0])
                time = float(nums[1])

                if time == 0:

                    return (
                        "Time cannot be zero."
                    )

                return (
                    "Speed = "
                    + str(
                        distance / time
                    )
                )

            if (
                "force" in t
                and len(nums) >= 2
            ):

                mass = float(nums[0])
                acceleration = float(nums[1])

                return (
                    "Force = "
                    + str(
                        mass * acceleration
                    )
                    + " N"
                )

            if (
                "kinetic energy" in t
                and len(nums) >= 2
            ):

                mass = float(nums[0])
                velocity = float(nums[1])

                energy = (
                    0.5
                    * mass
                    * velocity ** 2
                )

                return (
                    "Kinetic Energy = "
                    + str(energy)
                    + " J"
                )

            if (
                "potential energy" in t
                and len(nums) >= 3
            ):

                mass = float(nums[0])
                gravity = float(nums[1])
                height = float(nums[2])

                energy = (
                    mass
                    * gravity
                    * height
                )

                return (
                    "Potential Energy = "
                    + str(energy)
                    + " J"
                )

        except Exception:

            return None

        return None


# ============================================================
# CHEMISTRY
# ============================================================

class ChemistryEngine:

    @staticmethod
    def solve(text):

        t = text.lower()

        answers = {

            "h2o":
                "H2O is water.",

            "co2":
                "CO2 is carbon dioxide.",

            "nacl":
                "NaCl is sodium chloride, commonly called table salt.",

            "o2":
                "O2 is oxygen gas.",

            "h2":
                "H2 is hydrogen gas.",
        }

        for key, value in answers.items():

            if key in t:

                return value

        if (
            "atomic number of hydrogen"
            in t
        ):

            return (
                "Hydrogen has atomic number 1."
            )

        if (
            "atomic number of carbon"
            in t
        ):

            return (
                "Carbon has atomic number 6."
            )

        if (
            "atomic number of oxygen"
            in t
        ):

            return (
                "Oxygen has atomic number 8."
            )

        return None


# ============================================================
# PERIODIC TABLE
# ============================================================

class PeriodicTableEngine:

    ELEMENTS = {

        "h": ("Hydrogen", 1),
        "he": ("Helium", 2),
        "li": ("Lithium", 3),
        "be": ("Beryllium", 4),
        "b": ("Boron", 5),
        "c": ("Carbon", 6),
        "n": ("Nitrogen", 7),
        "o": ("Oxygen", 8),
        "f": ("Fluorine", 9),
        "ne": ("Neon", 10),
        "na": ("Sodium", 11),
        "mg": ("Magnesium", 12),
        "al": ("Aluminium", 13),
        "si": ("Silicon", 14),
        "p": ("Phosphorus", 15),
        "s": ("Sulfur", 16),
        "cl": ("Chlorine", 17),
        "ar": ("Argon", 18),
        "k": ("Potassium", 19),
        "ca": ("Calcium", 20),
    }

    @classmethod
    def solve(cls, text):

        t = text.lower()

        match = re.search(
            r"(?:element|symbol)\s+([a-z]{1,2})",
            t
        )

        if not match:

            return None

        symbol = match.group(1)

        if symbol in cls.ELEMENTS:

            name, number = cls.ELEMENTS[
                symbol
            ]

            return (
                name
                + " ("
                + symbol.title()
                + "), atomic number "
                + str(number)
                + "."
            )

        return None


# ============================================================
# BIOLOGY
# ============================================================

class BiologyEngine:

    @staticmethod
    def solve(text):

        t = text.lower()

        answers = {

            "photosynthesis":
                "Photosynthesis is the process by which green plants use light energy to make food from carbon dioxide and water.",

            "cell":
                "A cell is the basic structural and functional unit of life.",

            "dna":
                "DNA carries genetic information in living organisms.",

            "heart":
                "The heart pumps blood throughout the body.",

            "mitochondria":
                "Mitochondria are organelles involved in cellular energy production.",
        }

        for key, value in answers.items():

            if key in t:

                return value

        return None


# ============================================================
# GENERAL KNOWLEDGE
# ============================================================

class GeneralKnowledgeEngine:

    ANSWERS = {

        "capital of india":
            "The capital of India is New Delhi.",

        "capital of maharashtra":
            "The capital of Maharashtra is Mumbai.",

        "largest planet":
            "Jupiter is the largest planet in our Solar System.",

        "red planet":
            "Mars is commonly called the Red Planet.",

        "fastest land animal":
            "The cheetah is the fastest land animal.",

        "national animal of india":
            "The Bengal tiger is India's national animal.",

        "national bird of india":
            "The Indian peafowl is India's national bird.",
    }

    @classmethod
    def solve(cls, text):

        t = text.lower()

        for key, answer in cls.ANSWERS.items():

            if key in t:

                return answer

        return None


# ============================================================
# STUDY
# ============================================================

class StudyEngine:

    @staticmethod
    def solve(text):

        t = text.lower()

        if "study plan" in t:

            return (
                "Simple study plan:\n"
                "1. Choose one topic.\n"
                "2. Study for 25 minutes.\n"
                "3. Take a 5 minute break.\n"
                "4. Solve practice questions.\n"
                "5. Review mistakes."
            )

        if "how to study" in t:

            return (
                "Start with the concept, make short notes, "
                "practice questions, and revise using active recall."
            )

        if "photosynthesis" in t:

            return BiologyEngine.solve(t)

        return None


# ============================================================
# HOMEWORK
# ============================================================

class HomeworkEngine:

    @staticmethod
    def solve(text):

        t = text.lower()

        if "homework" in t:

            return (
                "Send the homework question here. "
                "I can explain the solution step by step."
            )

        return None


# ============================================================
# QUIZ
# ============================================================

class QuizEngine:

    @staticmethod
    def solve(text):

        t = text.lower()

        if "quiz" not in t:

            return None

        questions = [

            "Quiz: What is the capital of India?",

            "Quiz: Which planet is known as the Red Planet?",

            "Quiz: What is H2O commonly called?",

            "Quiz: Which gas do humans need for respiration?",
        ]

        return random.choice(
            questions
        )


# ============================================================
# FLASHCARD
# ============================================================

class FlashcardEngine:

    @staticmethod
    def solve(text):

        t = text.lower()

        if "flashcard" not in t:

            return None

        cards = [

            "Flashcard - Photosynthesis: Food-making process in green plants.",

            "Flashcard - DNA: Molecule carrying genetic information.",

            "Flashcard - Force: Mass multiplied by acceleration.",

            "Flashcard - H2O: Water.",
        ]

        return random.choice(
            cards
        )


# ============================================================
# LANGUAGE
# ============================================================

class LanguageEngine:

    @staticmethod
    def solve(text):

        t = text.lower()

        if t.startswith(
            "translate "
        ):

            phrase = text[
                10:
            ].strip()

            return (
                "Translation mode: "
                + phrase
                + "\n"
                "For high-quality translation, provide the target language too."
            )

        if "meaning of" in t:

            return (
                "Send the word and I can explain its meaning "
                "in simple language."
            )

        return None


# ============================================================
# REASONING
# ============================================================

class ReasoningEngine:

    @staticmethod
    def solve(text):

        t = text.lower()

        if "why is sky blue" in t:

            return (
                "The sky appears blue mainly because Earth's atmosphere "
                "scatters shorter blue wavelengths of sunlight more strongly."
            )

        if "riddle" in t:

            return (
                "Send me the riddle and I will try to solve it."
            )

        return None


# ============================================================
# CODING
# ============================================================

class CodingEngine:

    @staticmethod
    def solve(text):

        t = text.lower()

        coding_words = [

            "code",
            "coding",
            "program",
            "programming",
            "debug",
            "python",
            "java",
            "javascript",
            "kotlin",
            "html",
            "css",
            "sql"
        ]

        if not any(
            word in t
            for word in coding_words
        ):

            return None

        if (
            "hello world" in t
            and
            "python" in t
        ):

            return (
                "Python example:\n\n"
                "print('Hello World')"
            )

        if (
            "python" in t
            and
            "calculator" in t
        ):

            return (
                "Python calculator example:\n\n"
                "a = float(input('First number: '))\n"
                "b = float(input('Second number: '))\n"
                "print(a + b)"
            )

        return (
            "Coding mode active. "
            "Tell me the programming language and what you want to build, "
            "debug, explain, or convert."
        )


# ============================================================
# CREATIVE
# ============================================================

class CreativeEngine:

    @staticmethod
    def solve(text):

        t = text.lower()

        if "story" in t:

            return (
                "Creative story idea:\n"
                "A young creator discovers an offline AI assistant "
                "that learns from everyday conversations and helps "
                "turn small ideas into real projects."
            )

        if "caption" in t:

            return (
                "Caption idea:\n"
                "Dream big. Build quietly. Let the results speak."
            )

        if "script" in t:

            return (
                "Script structure:\n"
                "Hook -> Problem -> Journey -> Result -> Call to action."
            )

        if "idea" in t:

            return (
                "Content idea: Show the journey of building Layla "
                "from a simple Python project into an Android AI app."
            )

        return None


# ============================================================
# GAMING
# ============================================================

class GamingEngine:

    @staticmethod
    def solve(text):

        t = text.lower()

        if "free fire" in t:

            return (
                "For Free Fire content, I can help with tournament ideas, "
                "captions, practice plans, video concepts and scripts."
            )

        if (
            "gaming" in t
            or
            "game" in t
        ):

            return (
                "Gaming mode active. Tell me the game or the type "
                "of gaming content you want."
            )

        return None


# ============================================================
# VISION / MEDIA
# ============================================================

class VisionMediaEngine:

    @staticmethod
    def solve(text):

        t = text.lower()

        if any(
            word in t
            for word in [
                "photo",
                "image",
                "video",
                "file",
                "picture"
            ]
        ):

            return (
                "Media mode is ready. Use the Attach button to select "
                "a Photo, Video, or File."
            )

        return None


# ============================================================
# ATTACHMENT POPUP
# ============================================================

class AttachmentPopup(Popup):

    def __init__(
        self,
        callback,
        **kwargs
    ):

        super().__init__(
            title="Attach",
            size_hint=(0.8, None),
            height=dp(250),
            auto_dismiss=True,
            **kwargs
        )

        self.callback = callback

        layout = BoxLayout(
            orientation="vertical",
            spacing=dp(10),
            padding=dp(15)
        )

        photo = Button(
            text="Photo"
        )

        video = Button(
            text="Video"
        )

        file_button = Button(
            text="File"
        )

        photo.bind(
            on_release=lambda x:
            self.select("photo")
        )

        video.bind(
            on_release=lambda x:
            self.select("video")
        )

        file_button.bind(
            on_release=lambda x:
            self.select("file")
        )

        layout.add_widget(
            photo
        )

        layout.add_widget(
            video
        )

        layout.add_widget(
            file_button
        )

        self.content = layout

    def select(self, kind):

        self.dismiss()

        self.callback(kind)


# ============================================================
# CHAT BUBBLE
# ============================================================

class Bubble(BoxLayout):

    def __init__(
        self,
        text,
        user=False,
        **kwargs
    ):

        super().__init__(
            orientation="vertical",
            size_hint_y=None,
            padding=[
                dp(12),
                dp(9)
            ],
            **kwargs
        )

        self.user = user

        label = Label(
            text=text,
            color=(
                1,
                1,
                1,
                1
            ),
            font_size=dp(15),
            halign="left",
            valign="middle",
            size_hint_y=None
        )

        label.bind(
            texture_size=lambda instance, value:
            setattr(
                instance,
                "height",
                value[1]
            )
        )

        label.bind(
            width=lambda instance, value:
            setattr(
                instance,
                "text_size",
                (value, None)
            )
        )

        self.add_widget(
            label
        )

        self.height = max(
            dp(42),
            label.texture_size[1]
            + dp(18)
        )

        with self.canvas.before:

            if user:

                Color(
                    0.12,
                    0.35,
                    0.65,
                    1
                )

            else:

                Color(
                    0.15,
                    0.15,
                    0.18,
                    1
                )

            self.rect = RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[dp(14)]
            )

        self.bind(
            pos=self.update_rect,
            size=self.update_rect
        )

    def update_rect(self, *args):

        self.rect.pos = self.pos
        self.rect.size = self.size


# ============================================================
# LAYLA
# ============================================================

class Layla(BoxLayout):

    def __init__(
        self,
        **kwargs
    ):

        super().__init__(
            orientation="vertical",
            **kwargs
        )

        self.brain = OfflineBrain()

        self.voice_request_code = 1001

        self._voice_bound = False

        Window.clearcolor = (
            0.04,
            0.04,
            0.05,
            1
        )

        # ====================================================
        # HEADER
        # ====================================================

        header = BoxLayout(
            size_hint_y=None,
            height=dp(55),
            padding=[
                dp(15),
                dp(8)
            ]
        )

        title = Label(
            text="Layla",
            font_size=dp(22),
            bold=True,
            color=(
                1,
                1,
                1,
                1
            )
        )

        status = Label(
            text="Personal AI",
            font_size=dp(12),
            color=(
                0.65,
                0.65,
                0.68,
                1
            )
        )

        header.add_widget(
            title
        )

        header.add_widget(
            status
        )

        self.add_widget(
            header
        )

        # ====================================================
        # CHAT
        # ====================================================

        self.scroll = ScrollView(
            do_scroll_x=False,
            bar_width=dp(3)
        )

        self.chat = BoxLayout(
            orientation="vertical",
            spacing=dp(8),
            padding=[
                dp(10),
                dp(10)
            ],
            size_hint_y=None
        )

        self.chat.bind(
            minimum_height=self.chat.setter(
                "height"
            )
        )

        self.scroll.add_widget(
            self.chat
        )

        self.add_widget(
            self.scroll
        )

        # ====================================================
        # INPUT
        # ====================================================

        bottom = BoxLayout(
            size_hint_y=None,
            height=dp(60),
            spacing=dp(5),
            padding=[
                dp(6),
                dp(6)
            ]
        )

        attach = Button(
            text="Attach",
            size_hint_x=None,
            width=dp(75)
        )

        self.input_box = TextInput(
            hint_text="Talk to Layla...",
            multiline=False,
            font_size=dp(16),
            foreground_color=(
                1,
                1,
                1,
                1
            ),
            background_color=(
                0.12,
                0.12,
                0.14,
                1
            ),
            cursor_color=(
                1,
                1,
                1,
                1
            ),
            padding=[
                dp(10),
                dp(12)
            ]
        )

        mic = Button(
            text="Mic",
            size_hint_x=None,
            width=dp(55)
        )

        send = Button(
            text="Send",
            size_hint_x=None,
            width=dp(65)
        )

        attach.bind(
            on_release=self.open_attachment_menu
        )

        send.bind(
            on_release=self.send_message
        )

        self.input_box.bind(
            on_text_validate=self.send_message
        )

        mic.bind(
            on_release=self.voice_input
        )

        bottom.add_widget(
            attach
        )

        bottom.add_widget(
            self.input_box
        )

        bottom.add_widget(
            mic
        )

        bottom.add_widget(
            send
        )

        self.add_widget(
            bottom
        )

        self.add_bot_message(
            "Hello! Main Layla hoon. Mujhe kuch bhi pucho."
        )

    # ========================================================
    # ADD USER MESSAGE
    # ========================================================

    def add_user_message(self, text):

        bubble = Bubble(
            text,
            user=True
        )

        self.chat.add_widget(
            bubble
        )

        self.scroll_to_bottom()

    # ========================================================
    # ADD BOT MESSAGE
    # ========================================================

    def add_bot_message(self, text):

        bubble = Bubble(
            text,
            user=False
        )

        self.chat.add_widget(
            bubble
        )

        self.scroll_to_bottom()

    # ========================================================
    # SCROLL
    # ========================================================

    def scroll_to_bottom(self):

        from kivy.clock import Clock

        Clock.schedule_once(
            lambda dt:
            setattr(
                self.scroll,
                "scroll_y",
                0
            ),
            0.1
        )

    # ========================================================
    # SEND MESSAGE
    # ========================================================

    def send_message(self, *args):

        text = self.input_box.text.strip()

        if not text:

            return

        self.input_box.text = ""

        self.add_user_message(
            text
        )

        answer = self.process_message(
            text
        )

        self.add_bot_message(
            answer
        )

    # ========================================================
    # PROCESS MESSAGE
    # ========================================================

    def process_message(self, text):

        # Automatic learning
        self.brain.learn(
            text
        )

        # Normalize
        normalized = TextNormalizer.normalize(
            text
        )

        # Memory
        memory_answer = self.brain.answer(
            normalized
        )

        if memory_answer:

            return memory_answer

        # Conversation
        answer = ConversationEngine.solve(
            normalized
        )

        if answer:

            return answer

        # General knowledge
        answer = GeneralKnowledgeEngine.solve(
            normalized
        )

        if answer:

            return answer

        # Math
        answer = MathEngine.solve(
            normalized
        )

        if answer:

            return answer

        # Advanced math
        answer = AdvancedMathEngine.solve(
            normalized
        )

        if answer:

            return answer

        # Geometry
        answer = GeometryEngine.solve(
            normalized
        )

        if answer:

            return answer

        # Units
        answer = UnitEngine.solve(
            normalized
        )

        if answer:

            return answer

        # Physics
        answer = PhysicsEngine.solve(
            normalized
        )

        if answer:

            return answer

        # Chemistry
        answer = ChemistryEngine.solve(
            normalized
        )

        if answer:

            return answer

        # Periodic table
        answer = PeriodicTableEngine.solve(
            normalized
        )

        if answer:

            return answer

        # Biology
        answer = BiologyEngine.solve(
            normalized
        )

        if answer:

            return answer

        # Homework
        answer = HomeworkEngine.solve(
            normalized
        )

        if answer:

            return answer

        # Study
        answer = StudyEngine.solve(
            normalized
        )

        if answer:

            return answer

        # Quiz
        answer = QuizEngine.solve(
            normalized
        )

        if answer:

            return answer

        # Flashcards
        answer = FlashcardEngine.solve(
            normalized
        )

        if answer:

            return answer

        # Language
        answer = LanguageEngine.solve(
            normalized
        )

        if answer:

            return answer

        # Reasoning
        answer = ReasoningEngine.solve(
            normalized
        )

        if answer:

            return answer

        # Coding
        answer = CodingEngine.solve(
            normalized
        )

        if answer:

            return answer

        # Creative
        answer = CreativeEngine.solve(
            normalized
        )

        if answer:

            return answer

        # Gaming
        answer = GamingEngine.solve(
            normalized
        )

        if answer:

            return answer

        # Media
        answer = VisionMediaEngine.solve(
            normalized
        )

        if answer:

            return answer

        # Spell correction
        corrected = SpellCorrector.correct(
            normalized
        )

        if corrected != normalized:

            answer = ConversationEngine.solve(
                corrected
            )

            if answer:

                return answer

            answer = GeneralKnowledgeEngine.solve(
                corrected
            )

            if answer:

                return answer

            answer = MathEngine.solve(
                corrected
            )

            if answer:

                return answer

        return (
            "Main samajhne ki koshish kar rahi hoon. "
            "Thoda aur detail mein batao ki tum kya chahte ho."
        )

    # ========================================================
    # ATTACHMENT MENU
    # ========================================================

    def open_attachment_menu(self, *args):

        popup = AttachmentPopup(
            self.choose_attachment
        )

        popup.open()

    # ========================================================
    # CHOOSE ATTACHMENT
    # ========================================================

    def choose_attachment(self, kind):

        if filechooser is None:

            self.add_bot_message(
                "File picker available nahi hai. "
                "Build mein plyer dependency check karni hogi."
            )

            return

        try:

            filters = []

            if kind == "photo":

                filters = [
                    "*.png",
                    "*.jpg",
                    "*.jpeg",
                    "*.webp"
                ]

            elif kind == "video":

                filters = [
                    "*.mp4",
                    "*.mkv",
                    "*.avi",
                    "*.mov"
                ]

            selected = filechooser.open_file(
                multiple=False,
                filters=filters
            )

            if selected:

                self.attachment_selected(
                    kind,
                    selected[0]
                )

        except Exception as e:

            self.add_bot_message(
                "Attachment open nahi ho paya: "
                + str(e)
            )

    # ========================================================
    # ATTACHMENT SELECTED
    # ========================================================

    def attachment_selected(
        self,
        kind,
        path
    ):

        filename = os.path.basename(
            path
        )

        if kind == "photo":

            self.add_user_message(
                "Photo attached: "
                + filename
            )

            self.add_bot_message(
                "Photo select ho gayi. "
                "Actual image understanding ke liye "
                "vision/OCR model connect karna hoga."
            )

        elif kind == "video":

            self.add_user_message(
                "Video attached: "
                + filename
            )

            self.add_bot_message(
                "Video select ho gaya. "
                "Actual video analysis ke liye "
                "media/vision model connect karna hoga."
            )

        else:

            self.add_user_message(
                "File attached: "
                + filename
            )

            self.process_file(
                path
            )

    # ========================================================
    # BASIC FILE PROCESSING
    # ========================================================

    def process_file(self, path):

        try:

            extension = os.path.splitext(
                path
            )[1].lower()

            text_extensions = {

                ".txt",
                ".py",
                ".json",
                ".xml",
                ".html",
                ".css",
                ".js",
                ".java",
                ".kt",
                ".cpp",
                ".c",
                ".h",
                ".csv",
                ".md",
                ".yaml",
                ".yml"
            }

            if extension not in text_extensions:

                self.add_bot_message(
                    "File select ho gayi, lekin is file type "
                    + extension
                    + " ko abhi directly read nahi kar sakti."
                )

                return

            with open(
                path,
                "r",
                encoding="utf-8",
                errors="ignore"
            ) as f:

                content = f.read()

            if not content.strip():

                self.add_bot_message(
                    "File empty hai."
                )

                return

            max_chars = 6000

            preview = content[
                :max_chars
            ]

            if len(content) > max_chars:

                preview += (
                    "\n\n[File ka remaining content "
                    "preview limit ke karan nahi dikhaya gaya.]"
                )

            self.add_bot_message(
                "File read ho gayi.\n\n"
                + preview
            )

        except Exception as e:

            self.add_bot_message(
                "File process nahi ho payi: "
                + str(e)
            )

    # ========================================================
    # VOICE INPUT
    # ========================================================

    def voice_input(self, *args):

        try:

            from android import activity

            from jnius import autoclass

            self.voice_request_code = 1001

            # Activity result callback sirf ek baar bind karo
            if not getattr(
                self,
                "_voice_bound",
                False
            ):

                activity.bind(
                    on_activity_result=
                    self.on_voice_result
                )

                self._voice_bound = True

            Intent = autoclass(
                "android.content.Intent"
            )

            RecognizerIntent = autoclass(
                "android.speech.RecognizerIntent"
            )

            PythonActivity = autoclass(
                "org.kivy.android.PythonActivity"
            )

            intent = Intent(
                RecognizerIntent.ACTION_RECOGNIZE_SPEECH
            )

            intent.putExtra(
                RecognizerIntent.EXTRA_LANGUAGE_MODEL,
                RecognizerIntent.LANGUAGE_MODEL_FREE_FORM
            )

            intent.putExtra(
                RecognizerIntent.EXTRA_PROMPT,
                "Speak to Layla"
            )

            intent.putExtra(
                RecognizerIntent.EXTRA_MAX_RESULTS,
                1
            )

            PythonActivity.mActivity.startActivityForResult(
                intent,
                self.voice_request_code
            )

        except Exception as e:

            self.add_bot_message(
                "Voice input start nahi ho paya: "
                + str(e)
            )

    # ========================================================
    # VOICE RESULT
    # ========================================================

    def on_voice_result(
        self,
        request_code,
        result_code,
        intent
    ):

        if request_code != getattr(
            self,
            "voice_request_code",
            1001
        ):

            return

        try:

            from jnius import autoclass

            RESULT_OK = -1

            if (
                result_code != RESULT_OK
                or
                intent is None
            ):

                return

            RecognizerIntent = autoclass(
                "android.speech.RecognizerIntent"
            )

            results = intent.getStringArrayListExtra(
                RecognizerIntent.EXTRA_RESULTS
            )

            if (
                results
                and
                results.size() > 0
            ):

                text = str(
                    results.get(0)
                ).strip()

                if text:

                    self.input_box.text = text

                    self.input_box.focus = True

        except Exception as e:

            self.add_bot_message(
                "Voice result read nahi ho paya: "
                + str(e)
            )


# ============================================================
# APP
# ============================================================

class LaylaApp(App):

    def build(self):

        self.title = "Layla"

        return Layla()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    LaylaApp().run()