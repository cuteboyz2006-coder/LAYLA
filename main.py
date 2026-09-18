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


class Bubble(BoxLayout):
    def __init__(self, text, is_user=False, **kwargs):
        super().__init__(
            orientation="vertical",
            padding=[dp(16), dp(10), dp(16), dp(10)],
            spacing=dp(4),
            size_hint_y=None,
            **kwargs
        )

        self.is_user = is_user

        name = Label(
            text="You" if is_user else "Layla",
            color=(0.75, 0.55, 1, 1) if not is_user else (0.9, 0.9, 1, 1),
            font_size=dp(14),
            bold=True,
            size_hint_y=None,
            height=dp(22),
            halign="left",
        )

        message = Label(
            text=text,
            color=(0.95, 0.95, 1, 1),
            font_size=dp(17),
            size_hint_y=None,
            halign="left",
            valign="top",
        )

        message.text_size = (dp(290), None)

        def update_height(instance, value):
            instance.height = value[1] + dp(4)

        message.bind(texture_size=update_height)

        self.add_widget(name)
        self.add_widget(message)

        with self.canvas.before:
            if is_user:
                Color(0.20, 0.12, 0.45, 1)
            else:
                Color(0.07, 0.07, 0.13, 1)

            self.bg = RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[dp(22)]
            )

        self.bind(pos=self.update_bg, size=self.update_bg)

    def update_bg(self, *args):
        self.bg.pos = self.pos
        self.bg.size = self.size


class Layla(App):

    def build(self):

        root = BoxLayout(
            orientation="vertical",
            padding=[dp(14), dp(10), dp(14), dp(12)],
            spacing=dp(8)
        )

        # HEADER
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
            text="Personal AI Assistant",
            font_size=dp(17),
            color=(0.55, 0.52, 0.62, 1)
        )

        header.add_widget(title)
        header.add_widget(subtitle)

        root.add_widget(header)

        # CHAT
        scroll = ScrollView(
            do_scroll_x=False,
            bar_width=dp(3)
        )

        self.chat = BoxLayout(
            orientation="vertical",
            spacing=dp(12),
            size_hint_y=None
        )

        self.chat.bind(
            minimum_height=self.chat.setter("height")
        )

        scroll.add_widget(self.chat)
        root.add_widget(scroll)

        # INPUT AREA
        bottom = BoxLayout(
            size_hint_y=None,
            height=dp(58),
            spacing=dp(8)
        )

        self.message = TextInput(
            hint_text="Message Layla...",
            multiline=False,
            font_size=dp(17),
            padding=[dp(15), dp(15)],
            background_normal="",
            background_color=(0.07, 0.07, 0.13, 1),
            foreground_color=(0.95, 0.95, 1, 1),
            hint_text_color=(0.55, 0.55, 0.65, 1)
        )

        mic = Button(
            text="MIC",
            size_hint_x=None,
            width=dp(78),
            font_size=dp(15),
            background_normal="",
            background_color=(0.28, 0.16, 0.55, 1)
        )

        send = Button(
            text="SEND",
            size_hint_x=None,
            width=dp(88),
            font_size=dp(15),
            background_normal="",
            background_color=(0.38, 0.20, 0.75, 1)
        )

        send.bind(on_press=self.send_message)
        mic.bind(on_press=self.voice_input)

        bottom.add_widget(self.message)
        bottom.add_widget(mic)
        bottom.add_widget(send)

        root.add_widget(bottom)

        self.add_message(
            "Hello! Main Layla hoon.\n"
            "Main tumhari personal AI assistant hoon.\n"
            "Aaj tumse kya baat karni hai?",
            False
        )

        return root

    def add_message(self, text, is_user):

        row = BoxLayout(
            size_hint_y=None,
            padding=[dp(0), dp(0)],
        )

        bubble = Bubble(
            text=text,
            is_user=is_user,
            size_hint_x=None,
            width=dp(320)
        )

        if is_user:
            row.add_widget(Widget())
            row.add_widget(bubble)
        else:
            row.add_widget(bubble)
            row.add_widget(Widget())

        self.chat.add_widget(row)

    def ai_reply(self, message):

        msg = message.lower().strip()

        if msg in ["hi", "hello", "hey", "hii"]:
            return "Hey! 😊 Kaise ho?"

        if "tumhara naam" in msg or "your name" in msg:
            return "Mera naam Layla hai.\nMain tumhari personal AI assistant hoon."

        if "kaise ho" in msg or "how are you" in msg:
            return "Main bilkul ready hoon! 😄\nTum batao, kya karna hai?"

        if "kya kar sakti ho" in msg:
            return (
                "Main tumhare questions ka answer de sakti hoon, "
                "information provide kar sakti hoon aur voice mein baat kar sakti hoon."
            )

        if "thank" in msg or "thanks" in msg:
            return "You're welcome! 😊\nKoi aur baat?"

        return (
            "Maine tumhara message receive kar liya.\n"
            "Real AI brain next version mein connect karenge."
        )

    def send_message(self, instance):

        text = self.message.text.strip()

        if not text:
            return

        self.add_message(text, True)

        reply = self.ai_reply(text)

        self.add_message(reply, False)

        self.message.text = ""

    def voice_input(self, instance):

        # Existing working voice function
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

                self.add_message("Listening...", False)

            except Exception:
                self.add_message(
                    "Voice input start nahi ho paya.",
                    False
                )


if __name__ == "__main__":
    Layla().run()
            

            
