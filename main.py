import json
import os
import re
import ast
import operator
import math

from kivy.app import App
from kivy.core.window import Window
from kivy.metrics import dp
from kivy.graphics import Color, RoundedRectangle
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.uix.widget import Widget
from kivy.utils import platform


Window.clearcolor = (0.01, 0.005, 0.02, 1)


# =========================================================
# OFFLINE BRAIN
# =========================================================

class OfflineBrain:

    def __init__(self):

        self.file = os.path.join(
            App.get_running_app().user_data_dir,
            "knowledge.json"
        )

        self.knowledge = {}

        self.load()

    # -----------------------------------------------------

    def load(self):

        try:

            if os.path.exists(self.file):

                with open(
                    self.file,
                    "r",
                    encoding="utf-8"
                ) as f:

                    data = json.load(f)

                    if isinstance(data, dict):
                        self.knowledge = data
                    else:
                        self.knowledge = {}

        except Exception:

            self.knowledge = {}

    # -----------------------------------------------------

    def save(self):

        try:

            folder = os.path.dirname(self.file)

            if folder:
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
                    self.knowledge,
                    f,
                    ensure_ascii=False,
                    indent=2
                )

        except Exception:

            pass

    # -----------------------------------------------------

    def teach(self, question, answer):

        question = question.strip().lower()
        answer = answer.strip()

        if not question or not answer:
            return False

        self.knowledge[question] = answer

        self.save()

        return True

    # -----------------------------------------------------

    def teach_conversation(self, conversation):

        lines = conversation.splitlines()

        pending_question = None
        learned = 0

        for line in lines:

            line = line.strip()

            if not line:
                continue

            match_a = re.match(
                r"^(person\s*a|a|user)\s*:\s*(.+)$",
                line,
                re.IGNORECASE
            )

            if match_a:

                pending_question = match_a.group(2).strip()

                continue

            match_b = re.match(
                r"^(person\s*b|b|layla|assistant)\s*:\s*(.+)$",
                line,
                re.IGNORECASE
            )

            if match_b and pending_question:

                answer = match_b.group(2).strip()

                if answer:

                    self.knowledge[
                        pending_question.lower()
                    ] = answer

                    learned += 1

                    pending_question = None

        if learned > 0:
            self.save()

        return learned

    # -----------------------------------------------------

    def normalize(self, text):

        text = text.lower()

        text = re.sub(
            r"[^\w\s]",
            " ",
            text
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()

    # -----------------------------------------------------

    def find_answer(self, message):

        msg = self.normalize(message)

        if not msg:
            return None

        if msg in self.knowledge:
            return self.knowledge[msg]

        best_answer = None
        best_score = 0

        msg_words = set(
            msg.split()
        )

        for question, answer in self.knowledge.items():

            q_words = set(
                self.normalize(question).split()
            )

            if not q_words:
                continue

            common = msg_words.intersection(
                q_words
            )

            score = len(common) / len(q_words)

            if (
                score > best_score
                and score >= 0.45
            ):

                best_score = score
                best_answer = answer

        return best_answer

    # -----------------------------------------------------

    def count(self):

        return len(self.knowledge)


# =========================================================
# MATH ENGINE
# =========================================================

class MathEngine:

    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos
    }

    # -----------------------------------------------------

    def calculate(self, expression):

        expression = expression.strip()

        expression = expression.replace(
            "×",
            "*"
        )

        expression = expression.replace(
            "÷",
            "/"
        )

        expression = expression.replace(
            "^",
            "**"
        )

        expression = expression.replace(
            ",",
            ""
        )

        if not re.fullmatch(
            r"[\d\s\+\-\*\/\%\.\(\)]+",
            expression
        ):

            return None

        try:

            tree = ast.parse(
                expression,
                mode="eval"
            )

            value = self._evaluate(
                tree.body
            )

            if not math.isfinite(
                float(value)
            ):

                return None

            return value

        except Exception:

            return None

    # -----------------------------------------------------

    def _evaluate(self, node):

        if isinstance(
            node,
            ast.Constant
        ):

            if isinstance(
                node.value,
                (int, float)
            ):

                return node.value

            raise ValueError

        if isinstance(
            node,
            ast.BinOp
        ):

            if type(node.op) not in self.OPERATORS:
                raise ValueError

            left = self._evaluate(
                node.left
            )

            right = self._evaluate(
                node.right
            )

            if isinstance(
                node.op,
                ast.Pow
            ):

                if abs(right) > 100:
                    raise ValueError

            return self.OPERATORS[
                type(node.op)
            ](
                left,
                right
            )

        if isinstance(
            node,
            ast.UnaryOp
        ):

            if type(node.op) not in self.OPERATORS:
                raise ValueError

            return self.OPERATORS[
                type(node.op)
            ](
                self._evaluate(
                    node.operand
                )
            )

        raise ValueError

    # -----------------------------------------------------

    def format_number(self, value):

        if float(value).is_integer():

            return str(
                int(value)
            )

        return f"{value:.10g}"

    # -----------------------------------------------------

    def answer(self, message):

        text = message.lower().strip()

        # Percentage
        match = re.fullmatch(
            r"(\d+(?:\.\d+)?)\s*%\s*(?:of|ka)\s*(\d+(?:\.\d+)?)",
            text
        )

        if match:

            value = (
                float(match.group(1))
                * float(match.group(2))
                / 100
            )

            return self.format_number(
                value
            )

        expression = re.sub(
            r"\b(what is|calculate|solve|answer|equals)\b",
            "",
            text
        ).strip()

        expression = expression.replace(
            "×",
            "*"
        )

        expression = expression.replace(
            "÷",
            "/"
        )

        expression = expression.replace(
            "^",
            "**"
        )

        if not re.fullmatch(
            r"[\d\s\+\-\*\/\%\.\(\)\^×÷]+",
            expression
        ):

            return None

        value = self.calculate(
            expression
        )

        if value is None:
            return None

        return self.format_number(
            value
        )


# =========================================================
# PHYSICS ENGINE
# =========================================================

class PhysicsEngine:

    def numbers(self, text):

        return [
            float(x)
            for x in re.findall(
                r"(?<![a-z])\d+(?:\.\d+)?",
                text.lower()
            )
        ]

    # -----------------------------------------------------

    def fmt(self, value):

        if float(value).is_integer():

            return str(
                int(value)
            )

        return f"{value:.6g}"

    # -----------------------------------------------------

    def answer(self, message):

        m = message.lower().strip()

        nums = self.numbers(
            m
        )

        # Ohm's Law
        if (
            "ohm" in m
            or "ohm's law" in m
        ):

            return (
                "Ohm's Law:\n"
                "V = I × R\n\n"
                "V = Voltage\n"
                "I = Current\n"
                "R = Resistance"
            )

        # Newton 2nd Law
        if (
            "newton" in m
            and (
                "second law" in m
                or "2nd law" in m
            )
        ):

            return (
                "Newton's Second Law:\n"
                "F = m × a"
            )

        # Kinetic Energy
        if "kinetic energy" in m:

            return (
                "Kinetic Energy:\n"
                "KE = ½mv²"
            )

        # Potential Energy
        if "potential energy" in m:

            return (
                "Potential Energy:\n"
                "PE = mgh"
            )

        # Momentum
        if "momentum" in m:

            return (
                "Momentum:\n"
                "p = m × v"
            )

        # Force calculation
        if (
            "force" in m
            and (
                "mass" in m
                or "acceleration" in m
            )
            and len(nums) >= 2
        ):

            result = (
                nums[0]
                * nums[1]
            )

            return (
                "F = m × a\n"
                f"F = {self.fmt(nums[0])} × "
                f"{self.fmt(nums[1])}\n"
                f"F = {self.fmt(result)} N"
            )

        # Speed
        if (
            "speed" in m
            and "distance" in m
            and "time" in m
            and len(nums) >= 2
        ):

            result = (
                nums[0]
                / nums[1]
            )

            return (
                "Speed = Distance ÷ Time\n"
                f"Speed = {self.fmt(result)}"
            )

        # Distance
        if (
            "distance" in m
            and "speed" in m
            and "time" in m
            and len(nums) >= 2
        ):

            result = (
                nums[0]
                * nums[1]
            )

            return (
                "Distance = Speed × Time\n"
                f"Distance = {self.fmt(result)}"
            )

        return None


# =========================================================
# CHEMISTRY ENGINE
# =========================================================

class ChemistryEngine:

    elements = {

        "hydrogen": ("H", 1.008),

        "helium": ("He", 4.003),

        "carbon": ("C", 12.011),

        "nitrogen": ("N", 14.007),

        "oxygen": ("O", 15.999),

        "sodium": ("Na", 22.990),

        "magnesium": ("Mg", 24.305),

        "aluminium": ("Al", 26.982),

        "silicon": ("Si", 28.085),

        "phosphorus": ("P", 30.974),

        "sulfur": ("S", 32.06),

        "chlorine": ("Cl", 35.45),

        "potassium": ("K", 39.098),

        "calcium": ("Ca", 40.078),

        "iron": ("Fe", 55.845),

        "copper": ("Cu", 63.546),

        "zinc": ("Zn", 65.38),

        "silver": ("Ag", 107.868),

        "gold": ("Au", 196.967)

    }

    # -----------------------------------------------------

    def answer(self, message):

        m = message.lower().strip()

        if (
            "water" in m
            and (
                "formula" in m
                or "chemical" in m
            )
        ):

            return (
                "Water ka chemical formula: H₂O"
            )

        if (
            "carbon dioxide" in m
            and "formula" in m
        ):

            return (
                "Carbon dioxide ka formula: CO₂"
            )

        if (
            "mole" in m
            and "formula" in m
        ):

            return (
                "Moles ka formula:\n"
                "n = mass ÷ molar mass"
            )

        for name, data in self.elements.items():

            symbol, mass = data

            if (
                f"symbol of {name}" in m
                or m == name
            ):

                return (
                    f"{name.title()} = {symbol}"
                )

            if (
                name in m
                and (
                    "atomic mass" in m
                    or "atomic weight" in m
                )
            ):

                return (
                    f"{name.title()} ({symbol}) "
                    f"atomic mass ≈ {mass} u."
                )

        return None


# =========================================================
# UNIT CONVERTER
# =========================================================

class UnitEngine:

    factors = {

        "km": 1000,

        "m": 1,

        "cm": 0.01,

        "mm": 0.001,

        "kg": 1,

        "g": 0.001,

        "mg": 0.000001,

        "l": 1,

        "ml": 0.001

    }

    groups = [

        {
            "km",
            "m",
            "cm",
            "mm"
        },

        {
            "kg",
            "g",
            "mg"
        },

        {
            "l",
            "ml"
        }

    ]

    # -----------------------------------------------------

    def fmt(self, value):

        if float(value).is_integer():

            return str(
                int(value)
            )

        return f"{value:.8g}"

    # -----------------------------------------------------

    def answer(self, message):

        m = message.lower()

        m = m.replace(
            "→",
            " to "
        )

        m = re.sub(
            r"\s+",
            " ",
            m
        ).strip()

        match = re.search(
            r"(-?\d+(?:\.\d+)?)\s*"
            r"(km|m|cm|mm|kg|g|mg|l|ml)\s*"
            r"(?:to|in|into)\s*"
            r"(km|m|cm|mm|kg|g|mg|l|ml)",
            m
        )

        if match:

            value = float(
                match.group(1)
            )

            src = match.group(2)
            dst = match.group(3)

            compatible = any(
                src in group
                and dst in group
                for group in self.groups
            )

            if not compatible:

                return (
                    "Ye dono units "
                    "compatible nahi hain."
                )

            result = (
                value
                * self.factors[src]
                / self.factors[dst]
            )

            return (
                f"{self.fmt(result)} {dst}"
            )

        # Celsius to Fahrenheit
        c = re.search(
            r"(-?\d+(?:\.\d+)?)\s*"
            r"(?:°?\s*c)\s*to\s*"
            r"(?:°?\s*f|fahrenheit)",
            m
        )

        if c:

            value = float(
                c.group(1)
            )

            result = (
                value * 9 / 5
                + 32
            )

            return (
                f"{self.fmt(result)} °F"
            )

        # Fahrenheit to Celsius
        f = re.search(
            r"(-?\d+(?:\.\d+)?)\s*"
            r"(?:°?\s*f)\s*to\s*"
            r"(?:°?\s*c|celsius)",
            m
        )

        if f:

            value = float(
                f.group(1)
            )

            result = (
                value - 32
            ) * 5 / 9

            return (
                f"{self.fmt(result)} °C"
            )

        return None


# =========================================================
# STUDY ENGINE
# =========================================================

class StudyEngine:

    data = {

        "photosynthesis": (
            "Photosynthesis is the process by which "
            "green plants use light energy to make "
            "food from carbon dioxide and water."
        ),

        "cell": (
            "A cell is the basic structural and "
            "functional unit of life."
        ),

        "atom": (
            "An atom is the smallest unit of an "
            "element that retains its chemical "
            "properties."
        ),

        "newton's first law": (
            "An object remains at rest or in uniform "
            "motion unless acted upon by an external "
            "unbalanced force."
        )

    }

    # -----------------------------------------------------

    def answer(self, message):

        m = message.lower()

        for key, value in self.data.items():

            if (
                key in m
                and any(
                    word in m
                    for word in [
                        "what",
                        "explain",
                        "define",
                        "meaning",
                        "definition",
                        "kya"
                    ]
                )
            ):

                return value

        return None


# =========================================================
# QUIZ ENGINE
# =========================================================

class QuizEngine:

    questions = [

        (
            "Math",
            "What is 12 × 8?",
            "96"
        ),

        (
            "Physics",
            "What is the formula for force?",
            "F = m × a"
        ),

        (
            "Chemistry",
            "What is the formula of water?",
            "H2O"
        ),

        (
            "Biology",
            "What is the basic unit of life?",
            "Cell"
        )

    ]

    # -----------------------------------------------------

    def answer(self, message):

        m = message.lower()

        if "quiz" not in m:
            return None

        topic = None

        for name in [
            "math",
            "physics",
            "chemistry",
            "biology"
        ]:

            if name in m:

                topic = name.title()

                break

        pool = [
            q for q in self.questions
            if not topic
            or q[0] == topic
        ]

        if not pool:

            return (
                "Available quiz topics:\n"
                "Math, Physics, Chemistry, Biology"
            )

        subject, question, answer = pool[0]

        return (
            f"🎯 {subject} Quiz\n\n"
            f"Question: {question}\n\n"
            f"Answer: {answer}"
        )


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
            orientation="vertical",
            padding=[
                dp(16),
                dp(10),
                dp(16),
                dp(10)
            ],
            spacing=dp(4),
            size_hint_y=None,
            **kwargs
        )

        name = Label(
            text=(
                "You"
                if is_user
                else
                "Layla"
            ),
            color=(
                (0.75, 0.55, 1, 1)
                if not is_user
                else
                (0.9, 0.9, 1, 1)
            ),
            font_size=dp(14),
            bold=True,
            size_hint_y=None,
            height=dp(22),
            halign="left",
            valign="middle"
        )

        message = Label(
            text=text,
            color=(
                0.95,
                0.95,
                1,
                1
            ),
            font_size=dp(17),
            size_hint_y=None,
            halign="left",
            valign="top"
        )

        message.text_size = (
            dp(280),
            None
        )

        def update_message_height(
            instance,
            value
        ):

            message.height = value[1]

            self.height = (
                dp(10)
                + name.height
                + dp(4)
                + message.height
                + dp(10)
            )

        message.bind(
            texture_size=update_message_height
        )

        self.add_widget(name)
        self.add_widget(message)

        self.height = dp(50)

        with self.canvas.before:

            if is_user:

                Color(
                    0.20,
                    0.12,
                    0.45,
                    1
                )

            else:

                Color(
                    0.07,
                    0.07,
                    0.13,
                    1
                )

            self.bg = RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[
                    dp(22)
                ]
            )

        self.bind(
            pos=self.update_bg,
            size=self.update_bg
        )

    def update_bg(
        self,
        *args
    ):

        self.bg.pos = self.pos
        self.bg.size = self.size


# =========================================================
# LAYLA
# =========================================================

class Layla(App):

    def build(self):

        # Existing Brain
        self.brain = OfflineBrain()

        # New Engines
        self.math_engine = MathEngine()

        self.physics_engine = PhysicsEngine()

        self.chemistry_engine = ChemistryEngine()

        self.unit_engine = UnitEngine()

        self.study_engine = StudyEngine()

        self.quiz_engine = QuizEngine()

        # -------------------------------------------------

        root = BoxLayout(
            orientation="vertical",
            padding=[
                dp(14),
                dp(10),
                dp(14),
                dp(12)
            ],
            spacing=dp(8)
        )

        # -------------------------------------------------
        # HEADER
        # -------------------------------------------------

        header = BoxLayout(
            orientation="vertical",
            size_hint_y=None,
            height=dp(90)
        )

        header.add_widget(
            Label(
                text="LAYLA",
                font_size=dp(30),
                bold=True,
                color=(
                    0.85,
                    0.65,
                    1,
                    1
                )
            )
        )

        header.add_widget(
            Label(
                text="Personal AI • Offline Brain",
                font_size=dp(15),
                color=(
                    0.55,
                    0.52,
                    0.62,
                    1
                )
            )
        )

        root.add_widget(header)

        # -------------------------------------------------
        # CHAT
        # -------------------------------------------------

        scroll = ScrollView(
            do_scroll_x=False,
            bar_width=dp(3)
        )

        self.scroll = scroll

        self.chat = BoxLayout(
            orientation="vertical",
            spacing=dp(12),
            size_hint_y=None
        )

        self.chat.bind(
            minimum_height=self.chat.setter(
                "height"
            )
        )

        scroll.add_widget(
            self.chat
        )

        root.add_widget(
            scroll
        )

        # -------------------------------------------------
        # BUTTONS
        # -------------------------------------------------

        teach_row = BoxLayout(
            size_hint_y=None,
            height=dp(42),
            spacing=dp(7)
        )

        teach_button = Button(
            text="🧠 TEACH",
            font_size=dp(14),
            background_normal="",
            background_color=(
                0.25,
                0.12,
                0.50,
                1
            )
        )

        brain_button = Button(
            text="BRAIN",
            font_size=dp(14),
            background_normal="",
            background_color=(
                0.15,
                0.10,
                0.30,
                1
            )
        )

        teach_button.bind(
            on_press=self.teach_mode
        )

        brain_button.bind(
            on_press=self.show_brain
        )

        teach_row.add_widget(
            teach_button
        )

        teach_row.add_widget(
            brain_button
        )

        root.add_widget(
            teach_row
        )

        # -------------------------------------------------
        # INPUT
        # -------------------------------------------------

        bottom = BoxLayout(
            size_hint_y=None,
            height=dp(58),
            spacing=dp(8)
        )

        self.message = TextInput(
            hint_text="Message Layla...",
            multiline=False,
            font_size=dp(17),
            padding=[
                dp(15),
                dp(15)
            ],
            background_normal="",
            background_color=(
                0.07,
                0.07,
                0.13,
                1
            ),
            foreground_color=(
                0.95,
                0.95,
                1,
                1
            ),
            hint_text_color=(
                0.55,
                0.55,
                0.65,
                1
            )
        )

        mic = Button(
            text="MIC",
            size_hint_x=None,
            width=dp(70),
            font_size=dp(14),
            background_normal="",
            background_color=(
                0.28,
                0.16,
                0.55,
                1
            )
        )

        send = Button(
            text="SEND",
            size_hint_x=None,
            width=dp(78),
            font_size=dp(14),
            background_normal="",
            background_color=(
                0.38,
                0.20,
                0.75,
                1
            )
        )

        send.bind(
            on_press=self.send_message
        )

        mic.bind(
            on_press=self.voice_input
        )

        bottom.add_widget(
            self.message
        )

        bottom.add_widget(
            mic
        )

        bottom.add_widget(
            send
        )

        root.add_widget(
            bottom
        )

        # -------------------------------------------------
        # WELCOME
        # -------------------------------------------------

        self.add_message(
            "Hello! Main Layla hoon. 🧠\n\n"
            "Main tumse seekh sakti hoon.\n"
            "Math, Physics, Chemistry, Units, "
            "Study aur Quiz engines ready hain.",
            False
        )

        return root

    # =====================================================
    # ADD MESSAGE
    # =====================================================

    def add_message(
        self,
        text,
        is_user
    ):

        row = BoxLayout(
            size_hint_y=None
        )

        bubble = Bubble(
            text=text,
            is_user=is_user,
            size_hint_x=None,
            width=dp(320)
        )

        def update_row(
            instance,
            value
        ):

            row.height = bubble.height

        bubble.bind(
            height=update_row
        )

        row.height = bubble.height

        if is_user:

            row.add_widget(
                Widget()
            )

            row.add_widget(
                bubble
            )

        else:

            row.add_widget(
                bubble
            )

            row.add_widget(
                Widget()
            )

        self.chat.add_widget(
            row
        )

        self.scroll.scroll_y = 0

    # =====================================================
    # AI ROUTER
    # =====================================================

    def ai_reply(
        self,
        message
    ):

        msg = message.lower().strip()

        # Existing learned knowledge FIRST
        learned = self.brain.find_answer(
            message
        )

        if learned:

            return learned

        # -------------------------------------------------
        # MATH
        # -------------------------------------------------

        result = self.math_engine.answer(
            message
        )

        if result is not None:

            return (
                f"🧮 {result}"
            )

        # -------------------------------------------------
        # UNIT
        # -------------------------------------------------

        result = self.unit_engine.answer(
            message
        )

        if result is not None:

            return (
                f"📏 {result}"
            )

        # -------------------------------------------------
        # PHYSICS
        # -------------------------------------------------

        result = self.physics_engine.answer(
            message
        )

        if result is not None:

            return (
                f"⚛️ {result}"
            )

        # -------------------------------------------------
        # CHEMISTRY
        # -------------------------------------------------

        result = self.chemistry_engine.answer(
            message
        )

        if result is not None:

            return (
                f"🧪 {result}"
            )

        # -------------------------------------------------
        # STUDY
        # -------------------------------------------------

        result = self.study_engine.answer(
            message
        )

        if result is not None:

            return (
                f"📚 {result}"
            )

        # -------------------------------------------------
        # QUIZ
        # -------------------------------------------------

        result = self.quiz_engine.answer(
            message
        )

        if result is not None:

            return result

        # -------------------------------------------------
        # OLD BASIC RESPONSES
        # -------------------------------------------------

        if msg in [
            "hi",
            "hello",
            "hey",
            "hii",
            "helo"
        ]:

            return (
                "Hey! 😊\n"
                "Main Layla hoon."
            )

        if (
            "tumhara naam" in msg
            or "your name" in msg
        ):

            return (
                "Mera naam Layla hai.\n"
                "Main tumhari personal AI hoon."
            )

        if (
            "kaise ho" in msg
            or "how are you" in msg
        ):

            return (
                "Main ready hoon! 😄\n"
                "Mujhe kuch naya sikhao."
            )

        if (
            "brain" in msg
            and "kitna" in msg
        ):

            return (
                f"Mere brain me "
                f"{self.brain.count()} "
                f"learned memories hain."
            )

        if (
            "thank you" in msg
            or "thanks" in msg
        ):

            return (
                "You're welcome! 😊"
            )

        if msg in [
            "bye",
            "goodbye"
        ]:

            return (
                "Bye! 👋\n"
                "Phir milte hain."
            )

        return (
            "Mujhe iska answer abhi nahi pata. 🤔\n\n"
            "Tum mujhe sikha sakte ho:\n"
            "🧠 TEACH button dabao."
        )

    # =====================================================
    # SEND
    # =====================================================

    def send_message(
        self,
        instance
    ):

        text = self.message.text.strip()

        if not text:

            return

        self.add_message(
            text,
            True
        )

        # Teach
        if self.process_teaching(
            text
        ):

            self.message.text = ""

            return

        reply = self.ai_reply(
            text
        )

        self.add_message(
            reply,
            False
        )

        self.message.text = ""

    # =====================================================
    # TEACH MODE
    # =====================================================

    def teach_mode(
        self,
        instance
    ):

        self.add_message(
            "🧠 Teaching mode\n\n"
            "Normal teaching:\n"
            "teach: question = answer\n\n"
            "Example:\n"
            "teach: mera favourite game kya hai = Free Fire\n\n"
            "Conversation teaching format:\n"
            "teach conversation:\n"
            "Person A: Hello\n"
            "Person B: Hi!",
            False
        )

        self.message.text = ""

    # =====================================================
    # PROCESS TEACHING
    # =====================================================

    def process_teaching(
        self,
        text
    ):

        lower_text = text.lower().strip()

        # -------------------------------------------------
        # CONVERSATION TEACHING
        # -------------------------------------------------

        if lower_text.startswith(
            "teach conversation:"
        ):

            conversation = text[
                len("teach conversation:")
            :].strip()

            if not conversation:

                self.add_message(
                    "Conversation empty hai.\n\n"
                    "Example:\n"
                    "Person A: Hello\n"
                    "Person B: Hi!",
                    False
                )

                return True

            learned = self.brain.teach_conversation(
                conversation
            )

            if learned:

                self.add_message(
                    "🧠 Conversation Learned!\n\n"
                    f"{learned} responses "
                    "memory me save ho gaye.",
                    False
                )

            else:

                self.add_message(
                    "Conversation samajh nahi aayi.\n\n"
                    "Person A: Hello\n"
                    "Person B: Hi!",
                    False
                )

            return True

        # -------------------------------------------------
        # NORMAL TEACHING
        # -------------------------------------------------

        if not lower_text.startswith(
            "teach:"
        ):

            return False

        data = text[
            len("teach:")
        :].strip()

        if "=" not in data:

            self.add_message(
                "Teaching format galat hai.\n\n"
                "Example:\n"
                "teach: mera favourite game kya hai = Free Fire",
                False
            )

            return True

        question, answer = data.split(
            "=",
            1
        )

        question = question.strip()
        answer = answer.strip()

        if not question or not answer:

            self.add_message(
                "Question aur answer dono required hain.",
                False
            )

            return True

        success = self.brain.teach(
            question,
            answer
        )

        if success:

            self.add_message(
                "🧠 Learned!\n\n"
                f"Question: {question}\n"
                f"Answer: {answer}",
                False
            )

        return True

    # =====================================================
    # BRAIN STATUS
    # =====================================================

    def show_brain(
        self,
        instance
    ):

        count = self.brain.count()

        if count == 0:

            text = (
                "🧠 Layla Offline Brain\n\n"
                "Abhi brain empty hai.\n\n"
                "TEACH button se mujhe kuch sikhao."
            )

        else:

            text = (
                "🧠 Layla Offline Brain\n\n"
                f"Learned memories: {count}\n\n"
                "Knowledge phone ke app storage "
                "me save hai.\n\n"
                "App update karne par knowledge.json "
                "normally preserve rahega."
            )

        self.add_message(
            text,
            False
        )

    # =====================================================
    # MIC
    # =====================================================

    def voice_input(
        self,
        instance
    ):

        if platform == "android":

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

                self.add_message(
                    "🎙️ Listening...",
                    False
                )

            except Exception:

                self.add_message(
                    "Voice input start nahi ho paya.",
                    False
                )

        else:

            self.add_message(
                "Voice input Android device par available hai.",
                False
            )


# =========================================================
# START APP
# =========================================================

if __name__ == "__main__":

    Layla().run()