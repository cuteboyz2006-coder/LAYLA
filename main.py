import os

from kivy.app import App
from kivy.metrics import dp
from kivy.core.window import Window
from kivy.utils import platform
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.uix.widget import Widget


Window.clearcolor = (0.035, 0.025, 0.07, 1)


class MessageLabel(Label):
    def __init__(self, sender, message, **kwargs):
        super().__init__(**kwargs)

        self.text = f"{sender}\n{message}"
        self.markup = False
        self.font_size = dp(16)
        self.color = (0.94, 0.94, 0.98, 1)
        self.size_hint_y = None
        self.padding = (dp(14), dp(12))
        self.text_size = (dp(320), None)

        if sender == "Layla":
            self.text_size = (dp(320), None)
        else:
            self.text_size = (dp(320), None)

        self.bind(
            texture_size=self.update_height
        )

    def update_height(self, instance, value):
        self.height = value[1] + dp(18)


class Layla(App):

    def build(self):

        root = BoxLayout(
            orientation="vertical",
            padding=dp(10),
            spacing=dp(8)
        )

        # ---------- HEADER ----------

        header = BoxLayout(
            size_hint_y=None,
            height=dp(65),
            spacing=dp(10)
        )

        avatar_path = os.path.join(
            os.path.dirname(__file__),
            "avatar.png"
        )

        if os.path.exists(avatar_path):
            from kivy.uix.image import Image

            avatar = Image(
                source=avatar_path,
                size_hint_x=None,
                width=dp(50)
            )

            header.add_widget(avatar)

        title_box = BoxLayout(
            orientation="vertical"
        )

        title = Label(
            text="LAYLA",
            font_size=dp(22),
            bold=True,
            color=(0.95, 0.85, 1, 1),
            halign="left"
        )

        subtitle = Label(
            text="Personal AI Assistant",
            font_size=dp(12),
            color=(0.65, 0.62, 0.72, 1),
            halign="left"
        )

        title_box.add_widget(title)
        title_box.add_widget(subtitle)

        header.add_widget(title_box)

        root.add_widget(header)

        # ---------- CHAT ----------

        self.chat_box = BoxLayout(
            orientation="vertical",
            spacing=dp(10),
            size_hint_y=None,
            padding=(dp(5), dp(10))
        )

        self.chat_box.bind(
            minimum_height=self.chat_box.setter("height")
        )

        scroll = ScrollView(
            do_scroll_x=False
        )

        scroll.add_widget(self.chat_box)

        root.add_widget(scroll)

        self.scroll = scroll

        # ---------- INPUT ----------

        input_area = BoxLayout(
            size_hint_y=None,
            height=dp(55),
            spacing=dp(6)
        )

        self.message = TextInput(
            hint_text="Message Layla...",
            multiline=False,
            font_size=dp(16),
            padding=(dp(14), dp(14))
        )

        self.message.bind(
            on_text_validate=self.send_message
        )

        voice_button = Button(
            text="MIC",
            size_hint_x=None,
            width=dp(65),
            font_size=dp(13)
        )

        voice_button.bind(
            on_press=self.voice_input
        )

        send_button = Button(
            text="SEND",
            size_hint_x=None,
            width=dp(75),
            font_size=dp(13)
        )

        send_button.bind(
            on_press=self.send_message
        )

        input_area.add_widget(self.message)
        input_area.add_widget(voice_button)
        input_area.add_widget(send_button)

        root.add_widget(input_area)

        # ---------- WELCOME ----------

        self.add_message(
            "Layla",
            "Hello! Main Layla hoon. Main tumhari personal AI assistant hoon."
        )

        return root

    def add_message(self, sender, message):

        label = MessageLabel(
            sender,
            message
        )

        self.chat_box.add_widget(label)

        self.scroll_to_bottom()

    def scroll_to_bottom(self):

        def do_scroll(dt):
            self.scroll.scroll_y = 0

        from kivy.clock import Clock

        Clock.schedule_once(
            do_scroll,
            0.1
        )

    def ai_reply(self, message):

        msg = message.lower().strip()

        if msg in ["hi", "hello", "hey"]:
            return "Hello! Kaise ho? Main Layla hoon."

        if "tumhara naam" in msg:
            return "Mera naam Layla hai."

        if "kaise ho" in msg:
            return "Main bilkul ready hoon!"

        if "kya kar sakti ho" in msg:
            return (
                "Main tumhare questions ka answer de sakti hoon, "
                "voice mein baat kar sakti hoon aur AI assistant ke "
                "roop mein kaam kar sakti hoon."
            )

        return (
            "Maine tumhara message receive kar liya. "
            "Mera real AI brain next version mein connect karenge."
        )

    def send_message(self, instance):

        text = self.message.text.strip()

        if not text:
            return

        self.add_message(
            "You",
            text
        )

        reply = self.ai_reply(text)

        self.add_message(
            "Layla",
            reply
        )

        self.speak(reply)

        self.message.text = ""

    def voice_input(self, instance):

        if platform != "android":
            self.add_message(
                "Layla",
                "Voice input Android device par available hoga."
            )
            return

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
                RecognizerIntent.EXTRA_LANGUAGE,
                "hi-IN"
            )

            intent.putExtra(
                RecognizerIntent.EXTRA_PROMPT,
                "Layla se bolo..."
            )

            PythonActivity.mActivity.startActivityForResult(
                intent,
                1001
            )

            self.add_message(
                "Layla",
                "Listening..."
            )

        except Exception:
            self.add_message(
                "Layla",
                "Voice input start nahi ho paya."
            )

    def speak(self, text):

        if platform != "android":
            return

        try:

            from jnius import autoclass

            PythonActivity = autoclass(
                "org.kivy.android.PythonActivity"
            )

            TextToSpeech = autoclass(
                "android.speech.tts.TextToSpeech"
            )

            Locale = autoclass(
                "java.util.Locale"
            )

            activity = PythonActivity.mActivity

            tts = TextToSpeech(
                activity,
                None
            )

            tts.setLanguage(
                Locale("en", "IN")
            )

            tts.speak(
                text,
                TextToSpeech.QUEUE_FLUSH,
                None,
                "LAYLA"
            )

        except Exception:
            pass


if __name__ == "__main__":
    Layla().run()
