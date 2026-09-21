import json
import os
import re
import ast
import operator
import math
import random
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
from kivy.uix.popup import Popup


# =========================================================
# SAFE MATH ENGINE
# =========================================================

class MathEngine:

    operators = {
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

    def calculate(self, expression):

        expression = expression.lower().strip()

        expression = expression.replace("×", "*")
        expression = expression.replace("÷", "/")
        expression = expression.replace("^", "**")

        # Percentage
        percent = re.search(
            r"(\d+(?:\.\d+)?)\s*%\s*(?:of)\s*(\d+(?:\.\d+)?)",
            expression
        )

        if percent:
            a = float(percent.group(1))
            b = float(percent.group(2))
            result = a / 100 * b
            return f"{a}% of {b} = {result:g}"

        # Remove common words
        expression = re.sub(
            r"\b(what is|calculate|solve|answer|equals|equal to)\b",
            "",
            expression
        )

        expression = expression.strip()

        # Only allow arithmetic characters
        if not re.fullmatch(r"[0-9+\-*/().%\s]+", expression):
            return None

        try:
            tree = ast.parse(expression, mode="eval")
            result = self._eval(tree.body)

            if isinstance(result, float) and result.is_integer():
                result = int(result)

            return f"Answer: {result}"

        except Exception:
            return None

    def _eval(self, node):

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value

        if isinstance(node, ast.BinOp):
            if type(node.op) not in self.operators:
                raise ValueError()

            left = self._eval(node.left)
            right = self._eval(node.right)

            return self.operators[type(node.op)](left, right)

        if isinstance(node, ast.UnaryOp):
            if type(node.op) not in self.operators:
                raise ValueError()

            return self.operators[type(node.op)](
                self._eval(node.operand)
            )

        raise ValueError()


# =========================================================
# ADVANCED MATH ENGINE
# =========================================================

class AdvancedMathEngine:

    def solve(self, text):

        t = text.lower()

        # Square root
        m = re.search(r"(?:sqrt|square root of)\s*(\d+(?:\.\d+)?)", t)
        if m:
            n = float(m.group(1))
            return f"√{n:g} = {math.sqrt(n):g}"

        # Power
        m = re.search(
            r"(\d+(?:\.\d+)?)\s*(?:power|raised to|to the power of|\^)\s*(\d+(?:\.\d+)?)",
            t
        )
        if m:
            a = float(m.group(1))
            b = float(m.group(2))
            return f"{a:g}^{b:g} = {a ** b:g}"

        # Factorial
        m = re.search(r"(\d+)\s*(?:factorial|!)", t)
        if m:
            n = int(m.group(1))
            if n <= 100:
                return f"{n}! = {math.factorial(n)}"

        # Percentage
        m = re.search(
            r"(\d+(?:\.\d+)?)\s*%\s*(?:of)\s*(\d+(?:\.\d+)?)",
            t
        )
        if m:
            a = float(m.group(1))
            b = float(m.group(2))
            return f"{a:g}% of {b:g} = {a / 100 * b:g}"

        # Quadratic equation ax² + bx + c = 0
        m = re.search(
            r"([+-]?\d*\.?\d*)x\^2\s*([+-]\s*\d*\.?\d*)x\s*([+-]\s*\d*\.?\d*)\s*=\s*0",
            t
        )

        if m:
            try:
                a = float(m.group(1) or "1")
                b = float(m.group(2).replace(" ", "") or "0")
                c = float(m.group(3).replace(" ", "") or "0")

                d = b * b - 4 * a * c

                if d < 0:
                    return "Quadratic equation has no real roots."

                x1 = (-b + math.sqrt(d)) / (2 * a)
                x2 = (-b - math.sqrt(d)) / (2 * a)

                return f"Roots: x₁ = {x1:g}, x₂ = {x2:g}"

            except Exception:
                pass

        # Simple average
        if "average" in t or "mean" in t:
            nums = re.findall(r"\d+(?:\.\d+)?", t)

            if len(nums) >= 2:
                values = [float(x) for x in nums]
                avg = sum(values) / len(values)
                return f"Average = {avg:g}"

        # Trigonometry
        trig = {
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan
        }

        for name, func in trig.items():

            m = re.search(
                rf"{name}\s*(?:of)?\s*(\d+(?:\.\d+)?)",
                t
            )

            if m:
                degree = float(m.group(1))
                rad = math.radians(degree)

                try:
                    value = func(rad)
                    return f"{name}({degree:g}°) = {value:.6f}"
                except Exception:
                    pass

        return None


# =========================================================
# GEOMETRY ENGINE
# =========================================================

class GeometryEngine:

    def solve(self, text):

        t = text.lower()

        nums = [
            float(x)
            for x in re.findall(r"\d+(?:\.\d+)?", t)
        ]

        # Circle area
        if "circle" in t and "area" in t and nums:
            r = nums[0]
            return f"Circle area = πr² = {math.pi * r * r:.2f}"

        # Circle circumference
        if "circle" in t and (
            "circumference" in t or "perimeter" in t
        ) and nums:

            r = nums[0]
            return f"Circumference = 2πr = {2 * math.pi * r:.2f}"

        # Rectangle area
        if "rectangle" in t and "area" in t and len(nums) >= 2:
            l, w = nums[:2]
            return f"Rectangle area = {l * w:g}"

        # Rectangle perimeter
        if "rectangle" in t and "perimeter" in t and len(nums) >= 2:
            l, w = nums[:2]
            return f"Rectangle perimeter = {2 * (l + w):g}"

        # Square area
        if "square" in t and "area" in t and nums:
            s = nums[0]
            return f"Square area = {s * s:g}"

        # Square perimeter
        if "square" in t and "perimeter" in t and nums:
            s = nums[0]
            return f"Square perimeter = {4 * s:g}"

        # Triangle area
        if "triangle" in t and "area" in t and len(nums) >= 2:
            base, height = nums[:2]
            return f"Triangle area = ½ × base × height = {0.5 * base * height:g}"

        # Pythagoras
        if "pythagoras" in t or "pythagorean" in t:

            if len(nums) >= 2:
                a, b = nums[:2]
                c = math.sqrt(a * a + b * b)
                return f"Using Pythagoras: c = √(a²+b²) = {c:g}"

            return "Pythagoras theorem: a² + b² = c²"

        # Cube volume
        if "cube" in t and "volume" in t and nums:
            a = nums[0]
            return f"Cube volume = a³ = {a ** 3:g}"

        # Cuboid volume
        if "cuboid" in t and "volume" in t and len(nums) >= 3:
            l, w, h = nums[:3]
            return f"Cuboid volume = l × w × h = {l * w * h:g}"

        return None


# =========================================================
# PHYSICS ENGINE
# =========================================================

class PhysicsEngine:

    def solve(self, text):

        t = text.lower()

        if "ohm" in t:
            return "Ohm's Law: V = I × R"

        if "newton second law" in t or "newton's second law" in t:
            return "Newton's Second Law: F = m × a"

        if "kinetic energy" in t:
            return "Kinetic Energy: KE = ½mv²"

        if "potential energy" in t:
            return "Potential Energy: PE = mgh"

        if "momentum" in t:
            return "Momentum: p = m × v"

        nums = [
            float(x)
            for x in re.findall(r"\d+(?:\.\d+)?", t)
        ]

        if "force" in t and len(nums) >= 2:
            mass, acceleration = nums[:2]
            return f"Force = m × a = {mass * acceleration:g} N"

        if "speed" in t and "distance" in t and "time" in t:
            if len(nums) >= 2:
                distance, time = nums[:2]

                if time != 0:
                    return f"Speed = distance ÷ time = {distance / time:g}"

        if "distance" in t and "speed" in t and "time" in t:
            if len(nums) >= 2:
                speed, time = nums[:2]
                return f"Distance = speed × time = {speed * time:g}"

        return None


# =========================================================
# CHEMISTRY ENGINE
# =========================================================

class ChemistryEngine:

    elements = {
        "hydrogen": ("H", 1),
        "helium": ("He", 2),
        "lithium": ("Li", 3),
        "carbon": ("C", 6),
        "nitrogen": ("N", 7),
        "oxygen": ("O", 8),
        "fluorine": ("F", 9),
        "neon": ("Ne", 10),
        "sodium": ("Na", 11),
        "magnesium": ("Mg", 12),
        "aluminium": ("Al", 13),
        "aluminum": ("Al", 13),
        "silicon": ("Si", 14),
        "phosphorus": ("P", 15),
        "sulfur": ("S", 16),
        "chlorine": ("Cl", 17),
        "argon": ("Ar", 18),
        "potassium": ("K", 19),
        "calcium": ("Ca", 20),
        "iron": ("Fe", 26),
        "copper": ("Cu", 29),
        "zinc": ("Zn", 30),
        "silver": ("Ag", 47),
        "gold": ("Au", 79),
    }

    def solve(self, text):

        t = text.lower()

        if "water" in t and (
            "formula" in t or
            "chemical" in t
        ):
            return "Water chemical formula: H₂O"

        if "carbon dioxide" in t:
            return "Carbon dioxide formula: CO₂"

        if "oxygen" in t and "formula" in t:
            return "Oxygen molecule: O₂"

        if "mole" in t:
            return "Mole formula: n = mass ÷ molar mass"

        for name, data in self.elements.items():

            if name in t:

                symbol, atomic_number = data

                if "symbol" in t:
                    return f"{name.title()} symbol: {symbol}"

                if "atomic number" in t:
                    return f"{name.title()} atomic number: {atomic_number}"

        return None


# =========================================================
# PERIODIC TABLE ENGINE
# =========================================================

class PeriodicTableEngine:

    data = {
        "H": ("Hydrogen", 1),
        "He": ("Helium", 2),
        "Li": ("Lithium", 3),
        "Be": ("Beryllium", 4),
        "B": ("Boron", 5),
        "C": ("Carbon", 6),
        "N": ("Nitrogen", 7),
        "O": ("Oxygen", 8),
        "F": ("Fluorine", 9),
        "Ne": ("Neon", 10),
        "Na": ("Sodium", 11),
        "Mg": ("Magnesium", 12),
        "Al": ("Aluminium", 13),
        "Si": ("Silicon", 14),
        "P": ("Phosphorus", 15),
        "S": ("Sulfur", 16),
        "Cl": ("Chlorine", 17),
        "Ar": ("Argon", 18),
        "K": ("Potassium", 19),
        "Ca": ("Calcium", 20),
        "Fe": ("Iron", 26),
        "Cu": ("Copper", 29),
        "Zn": ("Zinc", 30),
        "Ag": ("Silver", 47),
        "Au": ("Gold", 79),
    }

    def solve(self, text):

        t = text.strip()

        m = re.search(
            r"(?:element|periodic table)\s*[:\-]?\s*([A-Z][a-z]?)$",
            t,
            re.IGNORECASE
        )

        if m:
            symbol = m.group(1)

            for key, value in self.data.items():

                if key.lower() == symbol.lower():

                    name, atomic = value

                    return (
                        f"{name}\n"
                        f"Symbol: {key}\n"
                        f"Atomic number: {atomic}"
                    )

        return None


# =========================================================
# UNIT ENGINE
# =========================================================

class UnitEngine:

    groups = {
        "length": {
            "km": 1000,
            "m": 1,
            "cm": 0.01,
            "mm": 0.001,
        },

        "mass": {
            "kg": 1000,
            "g": 1,
            "mg": 0.001,
        },

        "volume": {
            "l": 1,
            "ml": 0.001,
        }
    }

    def solve(self, text):

        t = text.lower().strip()

        m = re.search(
            r"(-?\d+(?:\.\d+)?)\s*([a-z]+)\s*(?:to|in|into)\s*([a-z]+)",
            t
        )

        if not m:
            return None

        value = float(m.group(1))
        source = m.group(2)
        target = m.group(3)

        for group in self.groups.values():

            if source in group and target in group:

                base = value * group[source]
                result = base / group[target]

                return f"{value:g} {source} = {result:g} {target}"

        # Celsius / Fahrenheit
        if source in ("c", "celsius") and target in ("f", "fahrenheit"):
            result = value * 9 / 5 + 32
            return f"{value:g}°C = {result:g}°F"

        if source in ("f", "fahrenheit") and target in ("c", "celsius"):
            result = (value - 32) * 5 / 9
            return f"{value:g}°F = {result:g}°C"

        return None


# =========================================================
# BIOLOGY ENGINE
# =========================================================

class BiologyEngine:

    knowledge = {

        "cell":
            "Cell is the basic structural and functional unit of life.",

        "photosynthesis":
            "Photosynthesis is the process by which green plants make food using sunlight, carbon dioxide and water.",

        "mitochondria":
            "Mitochondria are organelles involved in producing energy for the cell.",

        "nucleus":
            "The nucleus contains genetic material and controls many activities of the cell.",

        "dna":
            "DNA carries genetic information in living organisms.",

        "heart":
            "The heart pumps blood throughout the body.",

        "lungs":
            "The lungs help exchange oxygen and carbon dioxide.",

        "kidney":
            "Kidneys filter waste and extra water from the blood.",

        "chlorophyll":
            "Chlorophyll is the green pigment that helps plants absorb light for photosynthesis.",
    }

    def solve(self, text):

        t = text.lower()

        for key, answer in self.knowledge.items():

            if key in t:

                if any(word in t for word in [
                    "what",
                    "explain",
                    "define",
                    "meaning",
                    "kya",
                    "about"
                ]):
                    return answer

        return None


# =========================================================
# STUDY ENGINE
# =========================================================

class StudyEngine:

    topics = {

        "newton first law":
            "Newton's First Law: An object remains at rest or in uniform motion unless acted upon by an external force.",

        "atom":
            "An atom is the smallest unit of an element that retains its chemical identity.",

        "photosynthesis":
            "Plants use sunlight, water and carbon dioxide to produce food and release oxygen.",

        "gravity":
            "Gravity is the force of attraction between masses.",

        "ecosystem":
            "An ecosystem consists of living organisms interacting with each other and with their environment.",

        "democracy":
            "Democracy is a system of government in which people participate in choosing their representatives.",

        "computer":
            "A computer is an electronic device that processes data according to instructions.",

    }

    def solve(self, text):

        t = text.lower()

        for key, answer in self.topics.items():

            if key in t:

                if any(word in t for word in [
                    "what",
                    "explain",
                    "define",
                    "meaning",
                    "kya",
                    "tell me"
                ]):
                    return answer

        return None


# =========================================================
# HOMEWORK ENGINE
# =========================================================

class HomeworkEngine:

    def solve(self, text):

        t = text.lower()

        # Simple arithmetic homework
        if any(word in t for word in [
            "homework",
            "solve this",
            "solve question",
            "answer this"
        ]):

            expression = re.sub(
                r"\b(homework|solve this|solve question|answer this)\b",
                "",
                t
            ).strip()

            math_engine = MathEngine()
            answer = math_engine.calculate(expression)

            if answer:
                return (
                    "Homework Solution:\n\n"
                    f"Question: {expression}\n"
                    f"{answer}"
                )

        return None


# =========================================================
# QUIZ ENGINE
# =========================================================

class QuizEngine:

    questions = [

        {
            "topic": "math",
            "q": "What is 12 × 8?",
            "a": "96"
        },

        {
            "topic": "physics",
            "q": "What is Newton's Second Law?",
            "a": "F = m × a"
        },

        {
            "topic": "chemistry",
            "q": "What is the chemical formula of water?",
            "a": "H₂O"
        },

        {
            "topic": "biology",
            "q": "What is the basic unit of life?",
            "a": "Cell"
        },

        {
            "topic": "math",
            "q": "What is the square root of 144?",
            "a": "12"
        },

        {
            "topic": "physics",
            "q": "What is the formula of kinetic energy?",
            "a": "KE = ½mv²"
        },

        {
            "topic": "chemistry",
            "q": "What is the formula of carbon dioxide?",
            "a": "CO₂"
        },

    ]

    def solve(self, text):

        t = text.lower()

        if "quiz" not in t:
            return None

        selected = self.questions

        for topic in [
            "math",
            "physics",
            "chemistry",
            "biology"
        ]:

            if topic in t:
                selected = [
                    q for q in self.questions
                    if q["topic"] == topic
                ]

        if not selected:
            return "No quiz questions available."

        q = random.choice(selected)

        return (
            f"🎯 {q['topic'].title()} Quiz\n\n"
            f"Question: {q['q']}\n\n"
            f"Answer: {q['a']}"
        )


# =========================================================
# FLASHCARD ENGINE
# =========================================================

class FlashcardEngine:

    cards = [

        ("Newton's Second Law", "F = m × a"),

        ("Water formula", "H₂O"),

        ("Basic unit of life", "Cell"),

        ("Kinetic Energy", "KE = ½mv²"),

        ("Ohm's Law", "V = I × R"),

        ("Pythagoras theorem", "a² + b² = c²"),

        ("Photosynthesis", "Plants make food using light energy."),

        ("Momentum", "p = m × v"),

    ]

    def solve(self, text):

        if "flashcard" not in text.lower():
            return None

        question, answer = random.choice(self.cards)

        return (
            "🧠 Flashcard\n\n"
            f"Question: {question}\n"
            f"Answer: {answer}"
        )


# =========================================================
# LANGUAGE ENGINE
# =========================================================

class LanguageEngine:

    dictionary = {

        "hello": "नमस्ते / नमस्कार",

        "good morning": "सुप्रभात",

        "thank you": "धन्यवाद",

        "sorry": "माफ कीजिए",

        "water": "पानी",

        "school": "विद्यालय / स्कूल",

        "book": "किताब",

        "friend": "दोस्त / मित्र",

        "beautiful": "सुंदर",

        "happy": "खुश / प्रसन्न",

    }

    def solve(self, text):

        t = text.lower().strip()

        if (
            "meaning of" in t or
            "meaning" in t or
            "hindi meaning" in t or
            "english meaning" in t
        ):

            for word, meaning in self.dictionary.items():

                if word in t:
                    return f"{word.title()} = {meaning}"

        return None


# =========================================================
# REASONING ENGINE
# =========================================================

class ReasoningEngine:

    def solve(self, text):

        t = text.lower()

        if "odd one" in t or "odd-one" in t:

            nums = [
                int(x)
                for x in re.findall(r"\d+", t)
            ]

            if len(nums) >= 3:

                even = [x for x in nums if x % 2 == 0]
                odd = [x for x in nums if x % 2 != 0]

                if len(odd) == 1:
                    return f"Odd one out: {odd[0]}"

                if len(even) == 1:
                    return f"Odd one out: {even[0]}"

        # Number pattern
        if "pattern" in t:

            nums = [
                int(x)
                for x in re.findall(r"\d+", t)
            ]

            if len(nums) >= 3:

                differences = [
                    nums[i + 1] - nums[i]
                    for i in range(len(nums) - 1)
                ]

                if len(set(differences)) == 1:

                    next_number = nums[-1] + differences[0]

                    return (
                        f"Pattern difference = {differences[0]}\n"
                        f"Next number = {next_number}"
                    )

        return None


# =========================================================
# CODING ENGINE
# =========================================================

class CodingEngine:

    def solve(self, text):

        t = text.lower()

        if "python" in t and (
            "what" in t or
            "explain" in t or
            "meaning" in t
        ):
            return (
                "Python is a programming language known for "
                "simple syntax and wide use in automation, "
                "AI, data science and app development."
            )

        if "html" in t and (
            "what" in t or
            "explain" in t
        ):
            return (
                "HTML stands for HyperText Markup Language. "
                "It is used to structure web pages."
            )

        if "variable" in t and "programming" in t:
            return (
                "A variable is a named storage location "
                "used to hold a value in a program."
            )

        if "loop" in t and "python" in t:
            return (
                "Python commonly uses for loops and while loops "
                "to repeat code."
            )

        return None


# =========================================================
# GAMING ENGINE
# =========================================================

class GamingEngine:

    def solve(self, text):

        t = text.lower()

        if "free fire" in t:

            if "sensitivity" in t:
                return (
                    "Sensitivity is a control setting that changes "
                    "how quickly the aim/camera responds to touch. "
                    "The best setting depends on the device and "
                    "personal control preference."
                )

            if "tournament" in t:
                return (
                    "For a tournament, define the rules, "
                    "registration process, match timing, "
                    "room details and scoring system clearly."
                )

            if "headshot" in t:
                return (
                    "Headshot accuracy depends on aim control, "
                    "crosshair placement, movement and practice."
                )

        return None


# =========================================================
# OFFLINE BRAIN
# =========================================================

class OfflineBrain:

    def __init__(self):

        self.file = os.path.join(
            App.get_running_app().user_data_dir,
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

            os.makedirs(
                os.path.dirname(self.file),
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

    def teach(self, question, answer):

        question = question.strip().lower()

        if not question or not answer:
            return False

        self.data[question] = answer.strip()

        self.save()

        return True

    def teach_conversation(self, conversation):

        lines = conversation.splitlines()

        current_question = None
        count = 0

        for line in lines:

            line = line.strip()

            if not line:
                continue

            match = re.match(
                r"^(?:person\s*a|a|user)\s*[:\-]\s*(.+)$",
                line,
                re.IGNORECASE
            )

            if match:

                current_question = match.group(1).strip()

                continue

            match = re.match(
                r"^(?:person\s*b|b|layla|assistant)\s*[:\-]\s*(.+)$",
                line,
                re.IGNORECASE
            )

            if match and current_question:

                answer = match.group(1).strip()

                if self.teach(
                    current_question,
                    answer
                ):
                    count += 1

                current_question = None

        return count

    def find_answer(self, question):

        q = question.strip().lower()

        # Exact match
        if q in self.data:
            return self.data[q]

        # Similar word match
        q_words = set(
            re.findall(r"\w+", q)
        )

        if not q_words:
            return None

        best_answer = None
        best_score = 0

        for stored_q, answer in self.data.items():

            stored_words = set(
                re.findall(r"\w+", stored_q)
            )

            if not stored_words:
                continue

            common = len(
                q_words.intersection(stored_words)
            )

            score = common / max(
                len(q_words),
                len(stored_words)
            )

            if score > best_score:

                best_score = score
                best_answer = answer

        if best_score >= 0.45:
            return best_answer

        return None

    def count(self):
        return len(self.data)


# =========================================================
# CHAT BUBBLE
# =========================================================

class Bubble(BoxLayout):

    def __init__(
        self,
        text,
        is_user=False,
        **kwargs
    ):

        super().__init__(
            orientation="horizontal",
            size_hint_y=None,
            padding=[
                dp(12),
                dp(8),
                dp(12),
                dp(8)
            ],
            **kwargs
        )

        self.label = Label(
            text=text,
            color=(1, 1, 1, 1),
            halign="left",
            valign="middle",
            size_hint_y=None,
            text_size=(None, None)
        )

        self.label.bind(
            texture_size=self.update_height
        )

        self.add_widget(self.label)

        with self.canvas.before:

            Color(
                0.12,
                0.12,
                0.12,
                1
            )

            self.rect = RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[dp(12)]
            )

        self.bind(
            pos=self.update_rect,
            size=self.update_rect
        )

    def update_height(self, instance, size):

        self.label.text_size = (
            dp(270),
            None
        )

        self.height = (
            self.label.texture_size[1]
            + dp(16)
        )

    def update_rect(self, *args):

        self.rect.pos = self.pos
        self.rect.size = self.size


# =========================================================
# MAIN APP
# =========================================================

class Layla(App):

    def build(self):

        Window.clearcolor = (
            0.02,
            0.02,
            0.02,
            1
        )

        self.brain = OfflineBrain()

        self.math_engine = MathEngine()
        self.advanced_math = AdvancedMathEngine()
        self.geometry_engine = GeometryEngine()
        self.physics_engine = PhysicsEngine()
        self.chemistry_engine = ChemistryEngine()
        self.periodic_engine = PeriodicTableEngine()
        self.unit_engine = UnitEngine()
        self.biology_engine = BiologyEngine()
        self.study_engine = StudyEngine()
        self.homework_engine = HomeworkEngine()
        self.quiz_engine = QuizEngine()
        self.flashcard_engine = FlashcardEngine()
        self.language_engine = LanguageEngine()
        self.reasoning_engine = ReasoningEngine()
        self.coding_engine = CodingEngine()
        self.gaming_engine = GamingEngine()

        root = BoxLayout(
            orientation="vertical",
            padding=dp(8),
            spacing=dp(6)
        )

        # Header
        header = Label(
            text="LAYLA",
            font_size=dp(24),
            bold=True,
            size_hint_y=None,
            height=dp(50),
            color=(1, 1, 1, 1)
        )

        root.add_widget(header)

        # Chat area
        self.scroll = ScrollView(
            do_scroll_x=False
        )

        self.chat = BoxLayout(
            orientation="vertical",
            spacing=dp(6),
            size_hint_y=None
        )

        self.chat.bind(
            minimum_height=self.chat.setter(
                "height"
            )
        )

        self.scroll.add_widget(self.chat)

        root.add_widget(self.scroll)

        # Input
        self.message = TextInput(
            hint_text="Ask Layla...",
            multiline=False,
            size_hint_y=None,
            height=dp(48),
            background_color=(
                0.08,
                0.08,
                0.08,
                1
            ),
            foreground_color=(
                1,
                1,
                1,
                1
            ),
            cursor_color=(
                1,
                1,
                1,
                1
            )
        )

        root.add_widget(self.message)

        # Buttons
        buttons = BoxLayout(
            size_hint_y=None,
            height=dp(48),
            spacing=dp(5)
        )

        teach_btn = Button(
            text="TEACH"
        )

        brain_btn = Button(
            text="BRAIN"
        )

        mic_btn = Button(
            text="MIC"
        )

        send_btn = Button(
            text="SEND"
        )

        teach_btn.bind(
            on_press=self.teach_button
        )

        brain_btn.bind(
            on_press=self.show_brain
        )

        mic_btn.bind(
            on_press=self.voice_input
        )

        send_btn.bind(
            on_press=self.send_message
        )

        buttons.add_widget(teach_btn)
        buttons.add_widget(brain_btn)
        buttons.add_widget(mic_btn)
        buttons.add_widget(send_btn)

        root.add_widget(buttons)

        self.add_layla_message(
            "Hello! I'm Layla 🧠\n"
            "You can teach me and ask Math, Physics, "
            "Chemistry, Biology and other questions."
        )

        return root

    # =====================================================
    # CHAT
    # =====================================================

    def add_user_message(self, text):

        bubble = Bubble(
            text,
            is_user=True
        )

        self.chat.add_widget(bubble)

        self.scroll_to_bottom()

    def add_layla_message(self, text):

        bubble = Bubble(
            text,
            is_user=False
        )

        self.chat.add_widget(bubble)

        self.scroll_to_bottom()

    def scroll_to_bottom(self):

        from kivy.clock import Clock

        Clock.schedule_once(
            lambda dt: setattr(
                self.scroll,
                "scroll_y",
                0
            ),
            0.1
        )

    # =====================================================
    # SEND
    # =====================================================

    def send_message(self, instance):

        text = self.message.text.strip()

        if not text:
            return

        self.add_user_message(text)

        self.message.text = ""

        # Teaching
        teaching_result = self.process_teaching(text)

        if teaching_result:
            self.add_layla_message(
                teaching_result
            )
            return

        reply = self.ai_reply(text)

        self.add_layla_message(reply)

    # =====================================================
    # AI ROUTER
    # =====================================================

    def ai_reply(self, text):

        t = text.lower().strip()

        # ---------------------------------------------
        # BRAIN FIRST
        # ---------------------------------------------

        learned = self.brain.find_answer(text)

        if learned:
            return learned

        # ---------------------------------------------
        # MATH
        # ---------------------------------------------

        answer = self.math_engine.calculate(text)

        if answer:
            return answer

        # ---------------------------------------------
        # ADVANCED MATH
        # ---------------------------------------------

        answer = self.advanced_math.solve(text)

        if answer:
            return answer

        # ---------------------------------------------
        # GEOMETRY
        # ---------------------------------------------

        answer = self.geometry_engine.solve(text)

        if answer:
            return answer

        # ---------------------------------------------
        # UNIT
        # ---------------------------------------------

        answer = self.unit_engine.solve(text)

        if answer:
            return answer

        # ---------------------------------------------
        # PHYSICS
        # ---------------------------------------------

        answer = self.physics_engine.solve(text)

        if answer:
            return answer

        # ---------------------------------------------
        # CHEMISTRY
        # ---------------------------------------------

        answer = self.chemistry_engine.solve(text)

        if answer:
            return answer

        # ---------------------------------------------
        # PERIODIC TABLE
        # ---------------------------------------------

        answer = self.periodic_engine.solve(text)

        if answer:
            return answer

        # ---------------------------------------------
        # BIOLOGY
        # ---------------------------------------------

        answer = self.biology_engine.solve(text)

        if answer:
            return answer

        # ---------------------------------------------
        # HOMEWORK
        # ---------------------------------------------

        answer = self.homework_engine.solve(text)

        if answer:
            return answer

        # ---------------------------------------------
        # STUDY
        # ---------------------------------------------

        answer = self.study_engine.solve(text)

        if answer:
            return answer

        # ---------------------------------------------
        # QUIZ
        # ---------------------------------------------

        answer = self.quiz_engine.solve(text)

        if answer:
            return answer

        # ---------------------------------------------
        # FLASHCARD
        # ---------------------------------------------

        answer = self.flashcard_engine.solve(text)

        if answer:
            return answer

        # ---------------------------------------------
        # LANGUAGE
        # ---------------------------------------------

        answer = self.language_engine.solve(text)

        if answer:
            return answer

        # ---------------------------------------------
        # REASONING
        # ---------------------------------------------

        answer = self.reasoning_engine.solve(text)

        if answer:
            return answer

        # ---------------------------------------------
        # CODING
        # ---------------------------------------------

        answer = self.coding_engine.solve(text)

        if answer:
            return answer

        # ---------------------------------------------
        # GAMING
        # ---------------------------------------------

        answer = self.gaming_engine.solve(text)

        if answer:
            return answer

        # ---------------------------------------------
        # BASIC CHAT
        # ---------------------------------------------

        if any(
            x in t
            for x in [
                "hello",
                "hi",
                "hey",
                "hii"
            ]
        ):
            return "Hello! 👋 I'm Layla."

        if "how are you" in t:
            return "I'm ready to help you. 🧠"

        if "your name" in t:
            return "My name is Layla."

        if "who are you" in t:
            return (
                "I'm Layla, your personal offline AI assistant."
            )

        if "time" in t:

            return (
                "Current time: "
                + datetime.now().strftime("%I:%M %p")
            )

        if "date" in t:

            return (
                "Today's date: "
                + datetime.now().strftime("%d-%m-%Y")
            )

        return (
            "I don't know that yet. 🧠\n\n"
            "You can teach me using:\n"
            "teach: question = answer"
        )

    # =====================================================
    # TEACHING
    # =====================================================

    def process_teaching(self, text):

        # ---------------------------------------------
        # SIMPLE TEACH
        # ---------------------------------------------

        if text.lower().startswith("teach:"):

            content = text[6:].strip()

            if "=" not in content:

                return (
                    "Use this format:\n"
                    "teach: question = answer"
                )

            question, answer = content.split(
                "=",
                1
            )

            question = question.strip()
            answer = answer.strip()

            if self.brain.teach(
                question,
                answer
            ):

                return (
                    "✅ Learned!\n\n"
                    f"Question: {question}\n"
                    f"Answer: {answer}"
                )

            return "I couldn't save that lesson."

        # ---------------------------------------------
        # MULTI-LINE CONVERSATION
        # ---------------------------------------------

        if text.lower().startswith(
            "teach conversation:"
        ):

            conversation = text[
                len("teach conversation:")
            ].strip()

            count = self.brain.teach_conversation(
                conversation
            )

            return (
                f"✅ Conversation teaching complete.\n"
                f"Learned {count} conversation pair(s)."
            )

        return None

    # =====================================================
    # TEACH BUTTON
    # =====================================================

    def teach_button(self, instance):

        layout = BoxLayout(
            orientation="vertical",
            spacing=dp(8),
            padding=dp(8)
        )

        info = Label(
            text=(
                "Teach Layla\n\n"
                "Question = Answer\n\n"
                "Example:\n"
                "mera favourite game kya hai = Free Fire"
            )
        )

        question = TextInput(
            hint_text="Question",
            multiline=False
        )

        answer = TextInput(
            hint_text="Answer",
            multiline=True
        )

        save_btn = Button(
            text="SAVE"
        )

        layout.add_widget(info)
        layout.add_widget(question)
        layout.add_widget(answer)
        layout.add_widget(save_btn)

        popup = Popup(
            title="Teach Layla",
            content=layout,
            size_hint=(0.9, 0.75)
        )

        def save_lesson(instance):

            q = question.text.strip()
            a = answer.text.strip()

            if not q or not a:

                self.add_layla_message(
                    "Please enter both question and answer."
                )

                popup.dismiss()

                return

            if self.brain.teach(q, a):

                self.add_layla_message(
                    f"✅ Learned: {q}"
                )

            popup.dismiss()

        save_btn.bind(
            on_press=save_lesson
        )

        popup.open()

    # =====================================================
    # BRAIN
    # =====================================================

    def show_brain(self, instance):

        count = self.brain.count()

        self.add_layla_message(
            f"🧠 BRAIN\n\n"
            f"Learned knowledge: {count}\n"
            f"Storage: knowledge.json\n"
            f"Persistent: YES"
        )

    # =====================================================
    # MIC
    # =====================================================

    def voice_input(self, instance):

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

            self.add_layla_message(
                "🎤 Listening..."
            )

        except Exception:

            self.add_layla_message(
                "MIC is available when the Android speech "
                "recognition service is supported."
            )


# =========================================================
# START
# =========================================================

if __name__ == "__main__":
    Layla().run()