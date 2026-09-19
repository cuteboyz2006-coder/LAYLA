import json
import os
import re

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

    def load(self):

        try:

            if os.path.exists(self.file):

                with open(
                    self.file,
                    "r",
                    encoding="utf-8"
                ) as f:

                    self.knowledge = json.load(f)

        except Exception:

            self.knowledge = {}

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
                    self.knowledge,
                    f,
                    ensure_ascii=False,
                    indent=2
                )

        except Exception:

            pass

    def teach(self, question, answer):

        question = question.strip().lower()

        if not question or not answer.strip():

            return False

        self.knowledge[question] = answer.strip()

        self.save()

        return True

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

    def find_answer(self, message):

        msg = self.normalize(message)

        if not msg:

            return None

        # Exact match
        if msg in self.knowledge:

            return self.knowledge[msg]

        # Word matching
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

            if score > best_score and score >= 0.45:

                best_score = score
                best_answer = answer

        return best_answer

    def count(self):

        return len(self.knowledge)


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
            text="You" if is_user else "Layla",
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
            halign="left"
        )

        message = Label(
            text=text,
            color=(0.95, 0.95, 1, 1),
            font_size=dp(17),
            size_hint_y=None,
            halign="left",
            valign="top"
        )

        message.text_size = (
            dp(280),
            None
        )

        def update_height(instance, value):

            instance.height = value[1] + dp(8)

        message.bind(
            texture_size=update_height
        )

        self.add_widget(name)
        self.add_widget(message)

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
                radius=[dp(22)]
            )

        self.bind(
            pos=self.update_bg,
            size=self.update_bg
        )

    def update_bg(self, *args):

        self.bg.pos = self.pos
        self.bg.size = self.size


# =========================================================
# LAYLA APP
# =========================================================

class Layla(App):

    def build(self):

        self.brain = OfflineBrain()

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

        # =================================================
        # HEADER
        # =================================================

        header = BoxLayout(
            orientation="vertical",
            size_hint_y=None,
            height=dp(90)
        )

        title = Label(
            text="LAYLA",
            font_size=dp(30),
            bold=True,
            color=(0.85, 0.65, 1, 1)
        )

        subtitle = Label(
            text="Personal AI • Offline Brain",
            font_size=dp(15),
            color=(0.55, 0.52, 0.62, 1)
        )

        header.add_widget(title)
        header.add_widget(subtitle)

        root.add_widget(header)

        # =================================================
        # CHAT
        # =================================================

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

        scroll.add_widget(self.chat)

        root.add_widget(scroll)

        # =================================================
        # TEACH / BRAIN BUTTONS
        # =================================================

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

        root.add_widget(teach_row)

        # =================================================
        # INPUT
        # =================================================

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

        root.add_widget(bottom)

        # =================================================
        # WELCOME
        # =================================================

        self.add_message(
            "Hello! Main Layla hoon. 🧠\n\n"
            "Ab mere paas Offline Learning Brain hai.\n"
            "Tum mujhe cheezein sikha sakte ho.",
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

        def update_row(instance, value):

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

        self.chat.add_widget(row)

        self.scroll.scroll_y = 0

    # =====================================================
    # AI REPLY
    # =====================================================

    def ai_reply(self, message):

        msg = message.lower().strip()

        # First search learned knowledge
        learned = self.brain.find_answer(
            message
        )

        if learned:

            return learned

        # Basic built-in brain
        if msg in [
            "hi",
            "hello",
            "hey",
            "hii"
        ]:

            return (
                "Hey! 😊\n"
                "Main Layla hoon."
            )

        if (
            "tumhara naam" in msg
            or
            "your name" in msg
        ):

            return (
                "Mera naam Layla hai.\n"
                "Main tumhari personal AI hoon."
            )

        if (
            "kaise ho" in msg
            or
            "how are you" in msg
        ):

            return (
                "Main ready hoon! 😄\n"
                "Mujhe kuch naya sikhao."
            )

        if (
            "brain" in msg
            and
            "kitna" in msg
        ):

            return (
                f"Mere brain me "
                f"{self.brain.count()} "
                f"learned memories hain."
            )

        return (
            "Mujhe iska answer abhi nahi pata. 🤔\n\n"
            "Tum mujhe sikha sakte ho:\n"
            "🧠 TEACH button dabao."
        )

    # =====================================================
    # SEND MESSAGE
    # =====================================================

    def send_message(self, instance):

        text = self.message.text.strip()

        if not text:

            return

        self.add_message(
            text,
            True
        )

        # Teaching command
        if self.process_teaching(text):

            self.message.text = ""

            return

        reply = self.ai_reply(text)

        self.add_message(
            reply,
            False
        )

        self.message.text = ""

    # =====================================================
    # TEACH MODE
    # =====================================================

    def teach_mode(self, instance):

        self.add_message(
            "🧠 Teaching mode:\n\n"
            "Is format me message bhejo:\n\n"
            "teach: question = answer\n\n"
            "Example:\n"
            "teach: mera favourite game kya hai = Free Fire",
            False
        )

        self.message.text = ""

    # =====================================================
    # HANDLE TEACH COMMAND
    # =====================================================

    def process_teaching(
        self,
        text
    ):

        if not text.lower().startswith(
            "teach:"
        ):

            return False

        data = text[6:].strip()

        if "=" not in data:

            self.add_message(
                "Teaching format galat hai.\n\n"
                "Example:\n"
                "teach: mera naam kya hai = Cute Boy",
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

        self.brain.teach(
            question,
            answer
        )

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

    def show_brain(self, instance):

        count = self.brain.count()

        if count == 0:

            text = (
                "🧠 Brain empty hai.\n\n"
                "TEACH button se mujhe kuch sikhao."
            )

        else:

            text = (
                f"🧠 Layla Offline Brain\n\n"
                f"Learned knowledge: {count}\n\n"
                "Ye knowledge phone par save hai."
            )

        self.add_message(
            text,
            False
        )

    # =====================================================
    # VOICE INPUT
    # =====================================================

    def voice_input(self, instance):

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
