import json
import os
import re
import ast
import operator
import math
import random
import difflib
from datetime import datetime

from kivy.app import App
from kivy.core.window import Window
from kivy.metrics import dp
from kivy.graphics import Color, RoundedRectangle
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView

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
    def calc(cls, expression):
        expression = expression.strip()

        if len(expression) > 100:
            return None

        try:
            node = ast.parse(expression, mode="eval").body
            return cls._eval(node)
        except Exception:
            return None

    @classmethod
    def _eval(cls, node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError()

        if isinstance(node, ast.BinOp):
            if type(node.op) not in cls.OPS:
                raise ValueError()

            left = cls._eval(node.left)
            right = cls._eval(node.right)

            if isinstance(node.op, ast.Pow) and abs(right) > 20:
                raise ValueError()

            return cls.OPS[type(node.op)](left, right)

        if isinstance(node, ast.UnaryOp):
            if type(node.op) not in cls.OPS:
                raise ValueError()
            return cls.OPS[type(node.op)](cls._eval(node.operand))

        raise ValueError()


# ============================================================
# TEXT NORMALIZER
# ============================================================

class TextNormalizer:

    COMMON = {
        "helo": "hello",
        "helllo": "hello",
        "hii": "hi",
        "hiii": "hi",
        "hllo": "hello",
        "wat": "what",
        "wht": "what",
        "whats": "what is",
        "whts": "what is",
        "wher": "where",
        "wer": "where",
        "whre": "where",
        "hw": "how",
        "hwo": "how",
        "r": "are",
        "u": "you",
        "ur": "your",
        "yr": "your",
        "pls": "please",
        "plz": "please",
        "thx": "thanks",
        "tnx": "thanks",
        "bcoz": "because",
        "becoz": "because",
        "coz": "because",
        "frm": "from",
        "fr": "for",
        "abt": "about",
        "btw": "between",
        "wanna": "want to",
        "gonna": "going to",
        "dont": "do not",
        "cant": "cannot",
        "wont": "will not",
        "im": "i am",
        "iam": "i am",
        "ive": "i have",
        "idk": "i do not know",
        "kya": "kya",
        "h": "hai",
        "ha": "hai",
        "hu": "hoon",
        "ho": "ho",
        "kr": "kar",
        "kro": "karo",
        "btao": "batao",
        "btana": "batana",
        "acha": "achha",
        "accha": "achha",
        "kaise": "kaise",
        "kaisa": "kaisa",
        "mujhe": "mujhe",
        "mera": "mera",
        "meri": "meri",
        "tumhara": "tumhara",
        "tumhari": "tumhari",
    }

    @classmethod
    def normalize(cls, text):
        text = str(text).strip().lower()

        text = text.replace("×", "*")
        text = text.replace("÷", "/")
        text = text.replace("−", "-")
        text = text.replace("–", "-")
        text = text.replace("—", "-")
        text = text.replace("’", "'")

        text = re.sub(r"[!?;,]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        words = text.split()
        output = []

        for word in words:
            clean = re.sub(r"[^a-zA-Z0-9.+*/%^=-]", "", word)
            output.append(cls.COMMON.get(clean, clean))

        return " ".join(output)


# ============================================================
# SPELL CORRECTOR
# ============================================================

class SpellCorrector:

    WORDS = {
        "hello", "hi", "hey", "how", "are", "you", "what", "is",
        "your", "name", "calculate", "solve", "math", "physics",
        "chemistry", "biology", "photosynthesis", "mitochondria",
        "nucleus", "water", "formula", "element", "periodic",
        "table", "earth", "sun", "moon", "planet", "india",
        "capital", "computer", "programming", "python", "java",
        "javascript", "kotlin", "coding", "story", "write",
        "create", "game", "homework", "quiz", "flashcard",
        "gravity", "force", "energy", "speed", "distance",
        "circle", "area", "volume", "triangle", "square",
        "rectangle", "good", "morning", "night", "thanks",
        "thank", "bye", "please", "help", "learn", "remember",
        "favourite", "favorite", "food", "colour", "color",
        "weather", "time", "date", "free", "fire", "gaming"
    }

    @classmethod
    def correct_word(cls, word):
        if len(word) < 3 or word.isdigit():
            return word

        if word in cls.WORDS:
            return word

        matches = difflib.get_close_matches(
            word,
            cls.WORDS,
            n=1,
            cutoff=0.78
        )

        return matches[0] if matches else word

    @classmethod
    def correct(cls, text):
        parts = text.split()
        result = []

        for word in parts:
            # Never modify math expressions
            if re.search(r"[\d+\-*/%^().]", word):
                result.append(word)
            else:
                result.append(cls.correct_word(word))

        return " ".join(result)


# ============================================================
# CONVERSATION ENGINE
# ============================================================

class ConversationEngine:

    @staticmethod
    def solve(text):
        t = TextNormalizer.normalize(text)

        if re.search(r"\b(hello|hi|hey)\b", t):
            return random.choice([
                "Hello 😊",
                "Hi! Main Layla hoon. Kaise help karun?",
                "Hey! 😊 Kya karna hai?"
            ])

        if "how are you" in t or "kaise ho" in t:
            return "Main bilkul ready hoon 😊 Tum batao, kya karna hai?"

        if "what are you doing" in t:
            return "Main tumhari baat samajhne aur help karne ke liye ready hoon."

        if "what is your name" in t or "your name" in t:
            return "Mera naam Layla hai. 🤖"

        if "who are you" in t:
            return "Main Layla hoon — tumhari personal AI assistant."

        if "what can you do" in t:
            return (
                "Main maths, science, coding, general knowledge, "
                "creative work, conversation aur learning mein help kar sakti hoon."
            )

        if t in ("thanks", "thank you", "thank"):
            return "You're welcome 😊"

        if t in ("bye", "goodbye"):
            return "Bye! 😊 Phir milte hain."

        if "good morning" in t:
            return "Good morning ☀️"

        if "good night" in t:
            return "Good night 🌙"

        if t == "help":
            return "Bas normal language mein apna question bolo. Main khud suitable engine use karungi."

        return None


# ============================================================
# NATURAL CONVERSATION MEMORY
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
                with open(self.file, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
        except Exception:
            self.data = {}

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.file), exist_ok=True)

            with open(self.file, "w", encoding="utf-8") as f:
                json.dump(
                    self.data,
                    f,
                    ensure_ascii=False,
                    indent=2
                )
        except Exception:
            pass

    def learn(self, question, answer):
        q = TextNormalizer.normalize(question)

        if not q or not answer:
            return

        self.data[q] = answer
        self.save()

    def find_answer(self, question):
        q = TextNormalizer.normalize(question)

        if q in self.data:
            return self.data[q]

        best = None
        best_score = 0

        for key, value in self.data.items():
            score = difflib.SequenceMatcher(
                None,
                q,
                key
            ).ratio()

            q_words = set(q.split())
            k_words = set(key.split())

            if q_words and k_words:
                overlap = len(q_words & k_words) / max(
                    len(q_words),
                    len(k_words)
                )
                score = max(score, overlap)

            if score > best_score:
                best_score = score
                best = value

        if best_score >= 0.72:
            return best

        return None

    def automatic_learning(self, text):
        """
        Useful facts from normal conversation.
        No Teach button and no teach: command.
        """

        original = text.strip()

        patterns = [
            (
                r"my name is (.+)",
                "Your name is {}."
            ),
            (
                r"mera naam (.+) hai",
                "Tumhara naam {} hai."
            ),
            (
                r"my favourite game is (.+)",
                "Your favourite game is {}."
            ),
            (
                r"my favorite game is (.+)",
                "Your favourite game is {}."
            ),
            (
                r"mera favourite game (.+) hai",
                "Tumhara favourite game {} hai."
            ),
            (
                r"mera favorite game (.+) hai",
                "Tumhara favourite game {} hai."
            ),
            (
                r"my favourite color is (.+)",
                "Your favourite color is {}."
            ),
            (
                r"my favorite color is (.+)",
                "Your favourite color is {}."
            ),
            (
                r"my favorite food is (.+)",
                "Your favourite food is {}."
            ),
            (
                r"my favourite food is (.+)",
                "Your favourite food is {}."
            ),
            (
                r"mera favourite food (.+) hai",
                "Tumhara favourite food {} hai."
            ),
        ]

        low = TextNormalizer.normalize(original)

        for pattern, answer in patterns:
            match = re.fullmatch(pattern, low)

            if match:
                value = match.group(1).strip()

                if len(value) > 0 and len(value) < 80:
                    self.learn(original, answer.format(value))
                    return answer.format(value)

        return None


# ============================================================
# MATH ENGINE
# ============================================================

class MathEngine:

    @staticmethod
    def calculate(text):
        original = text.lower().strip()

        if "percent" in original or "%" in original:
            match = re.search(
                r"(\d+(?:\.\d+)?)\s*%\s*(?:of)?\s*(\d+(?:\.\d+)?)",
                original
            )

            if match:
                a = float(match.group(1))
                b = float(match.group(2))
                return str(round(a * b / 100, 8))

        t = original

        for phrase in [
            "what is",
            "calculate",
            "solve",
            "answer",
            "equals",
            "kitna hai",
            "kitna hoga"
        ]:
            t = t.replace(phrase, "")

        t = t.strip()
        t = t.replace("^", "**")

        if not re.fullmatch(
            r"[\d\s+\-*/().%*]+",
            t
        ):
            return None

        if not re.search(r"\d", t):
            return None

        result = SafeMath.calc(t)

        if result is None:
            return None

        if isinstance(result, float):
            if result.is_integer():
                return str(int(result))

            return str(round(result, 10))

        return str(result)


# ============================================================
# ADVANCED MATH
# ============================================================

class AdvancedMathEngine:

    @staticmethod
    def solve(text):
        t = text.lower()

        match = re.search(
            r"sqrt\s*(\d+(?:\.\d+)?)",
            t
        )

        if match:
            n = float(match.group(1))
            return str(round(math.sqrt(n), 8))

        match = re.search(
            r"factorial\s*(\d+)",
            t
        )

        if match:
            n = int(match.group(1))

            if n <= 100:
                return str(math.factorial(n))

        match = re.search(
            r"average\s+of\s+(.+)",
            t
        )

        if match:
            nums = re.findall(
                r"-?\d+(?:\.\d+)?",
                match.group(1)
            )

            if nums:
                values = [float(x) for x in nums]
                return str(round(sum(values) / len(values), 6))

        match = re.search(
            r"sin\s*\(?\s*(-?\d+(?:\.\d+)?)",
            t
        )

        if match:
            return str(
                round(
                    math.sin(math.radians(float(match.group(1)))),
                    8
                )
            )

        match = re.search(
            r"cos\s*\(?\s*(-?\d+(?:\.\d+)?)",
            t
        )

        if match:
            return str(
                round(
                    math.cos(math.radians(float(match.group(1)))),
                    8
                )
            )

        match = re.search(
            r"tan\s*\(?\s*(-?\d+(?:\.\d+)?)",
            t
        )

        if match:
            return str(
                round(
                    math.tan(math.radians(float(match.group(1)))),
                    8
                )
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

        if not nums:
            return None

        n = [float(x) for x in nums]

        if "circle" in t and "area" in t:
            r = n[0]
            return f"{round(math.pi * r * r, 6)} square units"

        if "circle" in t and (
            "circumference" in t or "perimeter" in t
        ):
            r = n[0]
            return f"{round(2 * math.pi * r, 6)} units"

        if "square" in t and "area" in t:
            return f"{round(n[0] * n[0], 6)} square units"

        if "square" in t and "perimeter" in t:
            return f"{round(4 * n[0], 6)} units"

        if "rectangle" in t and "area" in t and len(n) >= 2:
            return f"{round(n[0] * n[1], 6)} square units"

        if "rectangle" in t and "perimeter" in t and len(n) >= 2:
            return f"{round(2 * (n[0] + n[1]), 6)} units"

        if "triangle" in t and "area" in t and len(n) >= 2:
            return f"{round(0.5 * n[0] * n[1], 6)} square units"

        if "pythagoras" in t and len(n) >= 2:
            return f"{round(math.sqrt(n[0] ** 2 + n[1] ** 2), 6)}"

        if "cube" in t and "volume" in t:
            return f"{round(n[0] ** 3, 6)} cubic units"

        if "cuboid" in t and "volume" in t and len(n) >= 3:
            return f"{round(n[0] * n[1] * n[2], 6)} cubic units"

        return None


# ============================================================
# UNIT ENGINE
# ============================================================

class UnitEngine:

    @staticmethod
    def solve(text):
        t = text.lower()

        match = re.search(
            r"(-?\d+(?:\.\d+)?)\s*(km|m|cm|mm)\s*(?:to|in)\s*(km|m|cm|mm)",
            t
        )

        if match:
            value = float(match.group(1))
            a = match.group(2)
            b = match.group(3)

            meters = {
                "km": 1000,
                "m": 1,
                "cm": 0.01,
                "mm": 0.001
            }

            result = value * meters[a] / meters[b]

            return str(round(result, 8)) + " " + b

        match = re.search(
            r"(-?\d+(?:\.\d+)?)\s*(kg|g|mg)\s*(?:to|in)\s*(kg|g|mg)",
            t
        )

        if match:
            value = float(match.group(1))
            a = match.group(2)
            b = match.group(3)

            grams = {
                "kg": 1000,
                "g": 1,
                "mg": 0.001
            }

            result = value * grams[a] / grams[b]

            return str(round(result, 8)) + " " + b

        match = re.search(
            r"(-?\d+(?:\.\d+)?)\s*(c|celsius)\s*(?:to|in)\s*(f|fahrenheit)",
            t
        )

        if match:
            c = float(match.group(1))
            f = c * 9 / 5 + 32
            return str(round(f, 4)) + " °F"

        match = re.search(
            r"(-?\d+(?:\.\d+)?)\s*(f|fahrenheit)\s*(?:to|in)\s*(c|celsius)",
            t
        )

        if match:
            f = float(match.group(1))
            c = (f - 32) * 5 / 9
            return str(round(c, 4)) + " °C"

        return None


# ============================================================
# PHYSICS
# ============================================================

class PhysicsEngine:

    @staticmethod
    def solve(text):
        t = text.lower()

        nums = [
            float(x)
            for x in re.findall(
                r"-?\d+(?:\.\d+)?",
                t
            )
        ]

        if "ohm" in t and len(nums) >= 2:
            voltage = nums[0]
            resistance = nums[1]

            if resistance != 0:
                return f"Current = {round(voltage / resistance, 6)} A"

        if "force" in t and "mass" in t and "acceleration" in t:
            if len(nums) >= 2:
                return f"Force = {round(nums[0] * nums[1], 6)} N"

        if "kinetic energy" in t and len(nums) >= 2:
            m = nums[0]
            v = nums[1]
            return f"KE = {round(0.5 * m * v * v, 6)} J"

        if "potential energy" in t and len(nums) >= 2:
            m = nums[0]
            h = nums[1]
            return f"PE ≈ {round(m * 9.8 * h, 6)} J"

        if "momentum" in t and len(nums) >= 2:
            return f"Momentum = {round(nums[0] * nums[1], 6)} kg·m/s"

        if "speed" in t and "distance" in t and "time" in t:
            if len(nums) >= 2 and nums[1] != 0:
                return f"Speed = {round(nums[0] / nums[1], 6)}"

        return None


# ============================================================
# CHEMISTRY
# ============================================================

class ChemistryEngine:

    DATA = {
        "water": "H₂O",
        "carbon dioxide": "CO₂",
        "oxygen": "O₂",
        "hydrogen": "H₂",
        "nitrogen": "N₂",
        "sodium chloride": "NaCl",
        "salt": "NaCl",
        "ammonia": "NH₃",
        "methane": "CH₄",
    }

    @classmethod
    def solve(cls, text):
        t = text.lower()

        for name, formula in cls.DATA.items():
            if name in t and (
                "formula" in t or
                "chemical" in t or
                "symbol" in t
            ):
                return f"{name.title()} = {formula}"

        match = re.search(
            r"moles?\s*=\s*([\d.]+)\s*(?:mass)?\s*=?\s*([\d.]+)",
            t
        )

        if match:
            mass = float(match.group(1))
            molar = float(match.group(2))

            if molar != 0:
                return f"Moles = {round(mass / molar, 6)}"

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
        "fe": ("Iron", 26),
        "cu": ("Copper", 29),
        "zn": ("Zinc", 30),
        "ag": ("Silver", 47),
        "au": ("Gold", 79),
        "hg": ("Mercury", 80),
        "pb": ("Lead", 82),
        "u": ("Uranium", 92),
    }

    @classmethod
    def solve(cls, text):
        t = text.lower()

        match = re.search(
            r"(?:element|symbol)\s+([a-z]{1,2})\b",
            t
        )

        if match:
            symbol = match.group(1)

            if symbol in cls.ELEMENTS:
                name, number = cls.ELEMENTS[symbol]
                return f"{name} ({symbol.title()}), atomic number {number}"

        return None


# ============================================================
# BIOLOGY
# ============================================================

class BiologyEngine:

    FACTS = {
        "photosynthesis":
            "Photosynthesis is the process by which green plants use light energy to make food from carbon dioxide and water.",
        "mitochondria":
            "Mitochondria are organelles involved in producing usable cellular energy.",
        "nucleus":
            "The nucleus contains most of a cell's genetic material and helps control cell activities.",
        "dna":
            "DNA stores genetic information used by living organisms.",
        "chlorophyll":
            "Chlorophyll is the green pigment that absorbs light for photosynthesis.",
        "heart":
            "The heart is a muscular organ that pumps blood through the body.",
        "lungs":
            "The lungs are organs responsible for gas exchange, taking in oxygen and removing carbon dioxide.",
        "kidney":
            "The kidneys filter blood and help regulate water and dissolved substances in the body.",
        "cell":
            "A cell is the basic structural and functional unit of living organisms.",
    }

    @classmethod
    def solve(cls, text):
        t = TextNormalizer.normalize(text)

        for key, value in cls.FACTS.items():
            if key in t:
                return value

        return None


# ============================================================
# GENERAL KNOWLEDGE
# ============================================================

class GeneralKnowledgeEngine:

    FACTS = {
        "earth":
            "Earth is the third planet from the Sun.",
        "sun":
            "The Sun is a star at the center of our solar system.",
        "moon":
            "The Moon is Earth's natural satellite.",
        "solar system":
            "Our solar system contains the Sun and the objects that orbit it, including planets, dwarf planets, moons, asteroids and comets.",
        "mercury":
            "Mercury is the closest planet to the Sun.",
        "venus":
            "Venus is the second planet from the Sun.",
        "mars":
            "Mars is the fourth planet from the Sun.",
        "jupiter":
            "Jupiter is the largest planet in our solar system.",
        "saturn":
            "Saturn is well known for its prominent ring system.",
        "uranus":
            "Uranus is a planet with a strongly tilted axis.",
        "neptune":
            "Neptune is the eighth planet from the Sun.",
        "india":
            "India is a country in South Asia.",
        "capital of india":
            "The capital of India is New Delhi.",
        "continent":
            "Earth has seven commonly recognized continents: Africa, Antarctica, Asia, Europe, North America, Australia/Oceania and South America.",
        "oceans":
            "The five commonly recognized oceans are Pacific, Atlantic, Indian, Southern and Arctic.",
        "computer":
            "A computer is an electronic system that processes data according to instructions.",
        "cpu":
            "CPU stands for Central Processing Unit. It executes instructions and performs calculations.",
        "ram":
            "RAM is temporary working memory used by a computer while programs are running.",
        "internet":
            "The Internet is a global network of interconnected computer networks.",
        "html":
            "HTML is the standard markup language used to structure web pages.",
        "css":
            "CSS is used to describe the presentation and styling of web documents.",
        "python":
            "Python is a general-purpose programming language known for readable syntax and a large ecosystem.",
        "javascript":
            "JavaScript is a programming language widely used for interactive web applications and also for server-side and other software.",
        "gravity":
            "Gravity is the interaction that causes masses to attract one another.",
        "atom":
            "An atom is the basic unit of a chemical element.",
        "molecule":
            "A molecule consists of two or more atoms held together by chemical bonds.",
        "democracy":
            "Democracy is a system of government in which political power is exercised by the people, directly or through representatives.",
    }

    @classmethod
    def solve(cls, text):
        t = TextNormalizer.normalize(text)

        # More specific questions first
        for key, answer in cls.FACTS.items():
            if key in t:
                return answer

        return None


# ============================================================
# STUDY
# ============================================================

class StudyEngine:

    FACTS = {
        "newton first law":
            "Newton's first law says that an object remains at rest or in uniform motion unless acted on by a net external force.",
        "ecosystem":
            "An ecosystem includes living organisms and the non-living environment interacting together.",
        "gravity":
            "Gravity is the attraction between masses.",
        "atom":
            "An atom is the basic unit of a chemical element.",
        "photosynthesis":
            "Plants use light energy to make food from carbon dioxide and water during photosynthesis.",
    }

    @classmethod
    def solve(cls, text):
        t = TextNormalizer.normalize(text)

        for key, value in cls.FACTS.items():
            if key in t:
                return value

        return None


# ============================================================
# HOMEWORK
# ============================================================

class HomeworkEngine:

    @staticmethod
    def solve(text):
        t = TextNormalizer.normalize(text)

        triggers = [
            "homework",
            "solve this",
            "answer this",
            "question solve",
        ]

        if not any(x in t for x in triggers):
            return None

        result = MathEngine.calculate(text)

        if result:
            return f"Answer: {result}"

        return "Question bhejo, main usse solve karne ki koshish karungi."


# ============================================================
# QUIZ
# ============================================================

class QuizEngine:

    QUESTIONS = [
        (
            "What is 5 + 7?",
            ["10", "11", "12", "13"],
            "12"
        ),
        (
            "Which planet is known as the Red Planet?",
            ["Earth", "Mars", "Venus", "Jupiter"],
            "Mars"
        ),
        (
            "What is the chemical formula of water?",
            ["CO2", "H2O", "O2", "NaCl"],
            "H2O"
        ),
        (
            "Which organelle is associated with cellular energy production?",
            ["Nucleus", "Mitochondria", "Ribosome", "Cell wall"],
            "Mitochondria"
        ),
    ]

    @classmethod
    def solve(cls, text):
        t = TextNormalizer.normalize(text)

        if "quiz" not in t:
            return None

        q, options, answer = random.choice(cls.QUESTIONS)

        return (
            f"🧠 Quiz:\n{q}\n\n"
            f"A) {options[0]}\n"
            f"B) {options[1]}\n"
            f"C) {options[2]}\n"
            f"D) {options[3]}"
        )


# ============================================================
# FLASHCARDS
# ============================================================

class FlashcardEngine:

    CARDS = [
        ("Photosynthesis", "Plants use light to make food."),
        ("CPU", "Central Processing Unit."),
        ("DNA", "Stores genetic information."),
        ("Gravity", "Attraction between masses."),
        ("HTML", "Markup language for structuring web pages."),
        ("Python", "General-purpose programming language."),
    ]

    @classmethod
    def solve(cls, text):
        t = TextNormalizer.normalize(text)

        if "flashcard" not in t:
            return None

        front, back = random.choice(cls.CARDS)

        return f"📚 {front}\n\n{back}"


# ============================================================
# CODING ENGINE
# ============================================================

class CodingEngine:

    @staticmethod
    def solve(text):
        t = TextNormalizer.normalize(text)

        coding_words = [
            "python",
            "java",
            "javascript",
            "c++",
            "c programming",
            "c#",
            "kotlin",
            "swift",
            "php",
            "ruby",
            "rust",
            "go language",
            "sql",
            "html",
            "css",
            "coding",
            "programming",
            "code",
        ]

        if not any(x in t for x in coding_words):
            return None

        if "python" in t and "calculator" in t:
            return (
                "Python calculator example:\n\n"
                "a = float(input('First: '))\n"
                "b = float(input('Second: '))\n"
                "print(a + b)"
            )

        if "python" in t and "loop" in t:
            return (
                "Python loop example:\n\n"
                "for i in range(5):\n"
                "    print(i)"
            )

        if "html" in t and "page" in t:
            return (
                "<!DOCTYPE html>\n"
                "<html>\n"
                "<body>\n"
                "  <h1>Hello</h1>\n"
                "</body>\n"
                "</html>"
            )

        if "javascript" in t and "hello" in t:
            return "console.log('Hello World');"

        if "what is python" in t:
            return (
                "Python is a general-purpose programming language "
                "used for automation, web development, data work, AI and more."
            )

        if "what is html" in t:
            return "HTML is used to structure content on web pages."

        if "what is css" in t:
            return "CSS controls the styling and visual presentation of web pages."

        return (
            "💻 Coding mode: main programming concepts, code examples, "
            "debugging and explanations mein help kar sakti hoon."
        )


# ============================================================
# CREATIVE ENGINE
# ============================================================

class CreativeEngine:

    @staticmethod
    def solve(text):
        t = TextNormalizer.normalize(text)

        if not any(x in t for x in [
            "write a story",
            "make a story",
            "story",
            "creative",
            "script",
            "caption",
            "video idea",
            "content idea",
        ]):
            return None

        if "story" in t:
            return (
                "✨ Creative story idea:\n\n"
                "Ek ordinary student ko ek mysterious AI assistant "
                "milti hai jo sirf answers nahi deti, balki usse "
                "problems ko khud solve karna sikhati hai..."
            )

        if "script" in t:
            return (
                "🎬 Script structure:\n"
                "1. Hook\n"
                "2. Main idea\n"
                "3. Interesting moment\n"
                "4. Conclusion\n"
                "5. Call to action"
            )

        if "caption" in t:
            return "✨ Caption idea: Dream big. Build daily. Keep improving."

        return "🎨 Main story, script, caption aur content ideas create kar sakti hoon."

# ============================================================
# GAMING ENGINE
# ============================================================

class GamingEngine:

    @staticmethod
    def solve(text):
        t = TextNormalizer.normalize(text)

        if "free fire" in t:
            return (
                "🎮 Free Fire related questions, gaming ideas, "
                "tournament concepts aur basic game information mein help kar sakti hoon."
            )

        if "gaming" in t or "game" in t:
            return "🎮 Gaming mode active. Game ka naam aur question batao."

        return None


# ============================================================
# LANGUAGE ENGINE
# ============================================================

class LanguageEngine:

    DICT = {
        "hello": "नमस्ते",
        "water": "पानी",
        "food": "भोजन",
        "friend": "दोस्त",
        "school": "स्कूल",
        "book": "किताब",
        "computer": "कंप्यूटर",
        "game": "खेल",
        "love": "प्यार",
        "sun": "सूरज",
        "moon": "चाँद",
        "earth": "पृथ्वी",
    }

    @classmethod
    def solve(cls, text):
        t = TextNormalizer.normalize(text)

        match = re.search(
            r"(?:translate|meaning of)\s+([a-z]+)",
            t
        )

        if match:
            word = match.group(1)

            if word in cls.DICT:
                return f"{word} = {cls.DICT[word]}"

        return None


# ============================================================
# REASONING ENGINE
# ============================================================

class ReasoningEngine:

    @staticmethod
    def solve(text):
        t = TextNormalizer.normalize(text)

        if "odd one out" in t:
            return (
                "Odd-one-out question ke options bhejo, "
                "main reasoning ke saath answer karungi."
            )

        match = re.search(
            r"(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)",
            t
        )

        if match and "pattern" in t:
            nums = [
                int(match.group(i))
                for i in range(1, 5)
            ]

            d1 = nums[1] - nums[0]
            d2 = nums[2] - nums[1]
            d3 = nums[3] - nums[2]

            if d1 == d2 == d3:
                return f"Pattern is +{d1}. Next number = {nums[-1] + d1}"

        return None


# ============================================================
# VISION / MEDIA ENGINE
# ============================================================

class VisionMediaEngine:

    @staticmethod
    def solve(text):
        t = TextNormalizer.normalize(text)

        if any(x in t for x in [
            "photo",
            "image",
            "picture",
            "video",
            "camera",
        ]):
            return (
                "👁️ Vision module ready hai. Photo/video understanding "
                "ke liye actual local vision/OCR model connect karna hoga. "
                "Is architecture mein us module ko baad mein add kiya ja sakta hai."
            )

        return None


# ============================================================
# BUBBLE
# ============================================================

class Bubble(BoxLayout):

    def __init__(self, text, user=False, **kwargs):
        super().__init__(
            orientation="horizontal",
            size_hint_y=None,
            padding=(dp(12), dp(8)),
            spacing=dp(8),
            **kwargs
        )

        self.user = user

        label = Label(
            text=text,
            markup=True,
            halign="left",
            valign="middle",
            size_hint_y=None,
            text_size=(dp(285), None),
        )

        label.texture_update()

        label.height = max(
            dp(40),
            label.texture_size[1] + dp(18)
        )

        self.height = label.height + dp(16)

        with self.canvas.before:
            Color(
                0.10 if user else 0.16,
                0.10 if user else 0.16,
                0.10 if user else 0.16,
                1
            )

            self.rect = RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[dp(14)]
            )

        self.bind(
            pos=self._update_rect,
            size=self._update_rect
        )

        self.add_widget(label)

    def _update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size


# ============================================================
# MAIN APP
# ============================================================

class Layla(App):

    def build(self):
        Window.clearcolor = (0.035, 0.035, 0.035, 1)

        self.brain = OfflineBrain()

        self.conversation = ConversationEngine()
        self.math_engine = MathEngine()
        self.advanced_math = AdvancedMathEngine()
        self.geometry = GeometryEngine()
        self.physics = PhysicsEngine()
        self.chemistry = ChemistryEngine()
        self.periodic = PeriodicTableEngine()
        self.biology = BiologyEngine()
        self.general = GeneralKnowledgeEngine()
        self.study = StudyEngine()
        self.homework = HomeworkEngine()
        self.quiz = QuizEngine()
        self.flashcards = FlashcardEngine()
        self.coding = CodingEngine()
        self.creative = CreativeEngine()
        self.gaming = GamingEngine()
        self.language = LanguageEngine()
        self.reasoning = ReasoningEngine()
        self.vision = VisionMediaEngine()

        root = BoxLayout(
            orientation="vertical",
            padding=dp(8),
            spacing=dp(8)
        )

        # ---------------- HEADER ----------------

        header = BoxLayout(
            size_hint_y=None,
            height=dp(55),
            padding=(dp(8), dp(4))
        )

        title = Label(
            text="[b]LAYLA[/b]",
            markup=True,
            font_size=dp(22),
            halign="left",
            valign="middle"
        )

        subtitle = Label(
            text="Personal AI",
            font_size=dp(12),
            halign="right",
            valign="middle"
        )

        header.add_widget(title)
        header.add_widget(subtitle)

        root.add_widget(header)

        # ---------------- CHAT ----------------

        self.scroll = ScrollView(
            do_scroll_x=False
        )

        self.chat = BoxLayout(
            orientation="vertical",
            spacing=dp(8),
            size_hint_y=None,
            padding=(dp(4), dp(4))
        )

        self.chat.bind(
            minimum_height=self.chat.setter("height")
        )

        self.scroll.add_widget(self.chat)
        root.add_widget(self.scroll)

        # ---------------- INPUT ----------------

        bottom = BoxLayout(
            size_hint_y=None,
            height=dp(55),
            spacing=dp(6)
        )

        self.input = TextInput(
            hint_text="Talk to Layla...",
            multiline=False,
            size_hint_x=0.78,
            padding=(dp(12), dp(10)),
            font_size=dp(16)
        )

        self.input.bind(
            on_text_validate=self.send_message
        )

        send = Button(
            text="➤",
            size_hint_x=0.12,
            font_size=dp(20)
        )

        send.bind(
            on_release=self.send_message
        )

        mic = Button(
            text="🎤",
            size_hint_x=0.10
        )

        mic.bind(
            on_release=self.start_mic
        )

        bottom.add_widget(self.input)
        bottom.add_widget(send)
        bottom.add_widget(mic)

        root.add_widget(bottom)

        self.add_layla(
            "Hello! 👋 Main Layla hoon. "
            "Normal language mein mujhse baat karo."
        )

        return root

    # ========================================================
    # CHAT
    # ========================================================

    def add_user(self, text):
        self.chat.add_widget(
            Bubble(text, user=True)
        )
        self.scroll_to_bottom()

    def add_layla(self, text):
        self.chat.add_widget(
            Bubble(text, user=False)
        )
        self.scroll_to_bottom()

    def scroll_to_bottom(self):
        self.scroll.scroll_y = 0

    # ========================================================
    # MESSAGE
    # ========================================================

    def send_message(self, instance):
        text = self.input.text.strip()

        if not text:
            return

        self.input.text = ""

        self.add_user(text)

        answer = self.ai_reply(text)

        self.add_layla(answer)

    # ========================================================
    # AI ROUTER
    # ========================================================

    def ai_reply(self, original):

        # ----------------------------------------------------
        # Remove old Teach command completely
        # ----------------------------------------------------

        if original.lower().startswith("teach:"):
            return (
                "Ab Teach command ki zarurat nahi hai. "
                "Normal conversation se hi useful information learn karungi."
            )

        # ----------------------------------------------------
        # Natural learning
        # ----------------------------------------------------

        learned_now = self.brain.automatic_learning(original)

        if learned_now:
            return learned_now

        # ----------------------------------------------------
        # Normalize + spelling correction
        # ----------------------------------------------------

        normalized = TextNormalizer.normalize(original)
        corrected = SpellCorrector.correct(normalized)

        text = corrected

        # ----------------------------------------------------
        # Memory
        # ----------------------------------------------------

        memory_answer = self.brain.find_answer(text)

        if memory_answer:
            return memory_answer

        # ----------------------------------------------------
        # Conversation
        # ----------------------------------------------------

        answer = self.conversation.solve(text)

        if answer:
            return answer

        # ----------------------------------------------------
        # General Knowledge
        # ----------------------------------------------------

        answer = self.general.solve(text)

        if answer:
            return answer

        # ----------------------------------------------------
        # Math
        # ----------------------------------------------------

        answer = self.math_engine.calculate(text)

        if answer is not None:
            return f"Answer: {answer}"

        answer = self.advanced_math.solve(text)

        if answer:
            return f"Answer: {answer}"

        # ----------------------------------------------------
        # Geometry
        # ----------------------------------------------------

        answer = self.geometry.solve(text)

        if answer:
            return answer

        # ----------------------------------------------------
        # Units
        # ----------------------------------------------------

        answer = self.unit_engine_safe(text)

        if answer:
            return answer

        # ----------------------------------------------------
        # Physics
        # ----------------------------------------------------

        answer = self.physics.solve(text)

        if answer:
            return answer

        # ----------------------------------------------------
        # Chemistry
        # ----------------------------------------------------

        answer = self.chemistry.solve(text)

        if answer:
            return answer

        # ----------------------------------------------------
        # Periodic table
        # ----------------------------------------------------

        answer = self.periodic.solve(text)

        if answer:
            return answer

        # ----------------------------------------------------
        # Biology
        # ----------------------------------------------------

        answer = self.biology.solve(text)

        if answer:
            return answer

        # ----------------------------------------------------
        # Homework
        # ----------------------------------------------------

        answer = self.homework.solve(text)

        if answer:
            return answer

        # ----------------------------------------------------
        # Study
        # ----------------------------------------------------

        answer = self.study.solve(text)

        if answer:
            return answer

        # ----------------------------------------------------
        # Quiz
        # ----------------------------------------------------

        answer = self.quiz.solve(text)

        if answer:
            return answer

        # ----------------------------------------------------
        # Flashcards
        # ----------------------------------------------------

        answer = self.flashcards.solve(text)

        if answer:
            return answer

        # ----------------------------------------------------
        # Language
        # ----------------------------------------------------

        answer = self.language.solve(text)

        if answer:
            return answer

        # ----------------------------------------------------
        # Reasoning
        # ----------------------------------------------------

        answer = self.reasoning.solve(text)

        if answer:
            return answer

        # ----------------------------------------------------
        # Coding
        # ----------------------------------------------------

        answer = self.coding.solve(text)

        if answer:
            return answer

        # ----------------------------------------------------
        # Creative
        # ----------------------------------------------------

        answer = self.creative.solve(text)

        if answer:
            return answer

        # ----------------------------------------------------
        # Gaming
        # ----------------------------------------------------

        answer = self.gaming.solve(text)

        if answer:
            return answer

        # ----------------------------------------------------
        # Vision / Media
        # ----------------------------------------------------

        answer = self.vision.solve(text)

        if answer:
            return answer

        # ----------------------------------------------------
        # Spelling retry
        # ----------------------------------------------------

        if corrected != normalized:

            memory_answer = self.brain.find_answer(corrected)

            if memory_answer:
                return memory_answer

            general_answer = self.general.solve(corrected)

            if general_answer:
                return general_answer

            conversation_answer = self.conversation.solve(corrected)

            if conversation_answer:
                return conversation_answer

        # ----------------------------------------------------
        # Unknown
        # ----------------------------------------------------

        return (
            "Mujhe abhi iska exact answer nahi pata. "
            "Thoda aur detail mein batao, main samajhne ki koshish karungi. 😊"
        )

    def unit_engine_safe(self, text):
        return UnitEngine.solve(text)

    # ========================================================
    # MIC
    # ========================================================

    def start_mic(self, instance):
        try:
            from jnius import autoclass

            PythonActivity = autoclass(
                "org.kivy.android.PythonActivity"
            )

            Intent = autoclass(
                "android.content.Intent"
            )

            RecognizerIntent = autoclass(
                "android.speech.RecognizerIntent"
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

            PythonActivity.mActivity.startActivityForResult(
                intent,
                100
            )

        except Exception:
            self.add_layla(
                "🎤 Voice input Android environment mein available nahi hai."
            )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    Layla().run()