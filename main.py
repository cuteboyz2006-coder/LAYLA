import os

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.core.window import Window


Window.clearcolor = (0.04, 0.02, 0.08, 1)


class Layla(App):

    def build(self):
        root = BoxLayout(
            orientation="vertical",
            padding=10,
            spacing=8
        )

        header = BoxLayout(
            size_hint_y=None,
            height=90
        )

        avatar_path = os.path.join(
            os.path.dirname(__file__),
            "avatar.png"
        )

        avatar = Image(
            source=avatar_path,
            size_hint_x=None,
            width=75
        )

        title = Label(
            text="[b]LAYLA[/b]\nPersonal AI Assistant",
            markup=True,
            font_size=20
        )

        header.add_widget(avatar)
        header.add_widget(title)
        root.add_widget(header)

        self.chat = GridLayout(
            cols=1,
            spacing=8,
            size_hint_y=None
        )

        self.chat.bind(
            minimum_height=self.chat.setter("height")
        )

        scroll = ScrollView()
        scroll.add_widget(self.chat)
        root.add_widget(scroll)

        bottom = BoxLayout(
            size_hint_y=None,
            height=55,
            spacing=5
        )

        self.message = TextInput(
            hint_text="Layla se baat karo...",
            multiline=False
        )

        send = Button(
            text="SEND",
            size_hint_x=None,
            width=90
        )

        send.bind(on_press=self.send_message)

        bottom.add_widget(self.message)
        bottom.add_widget(send)
        root.add_widget(bottom)

        self.add_message(
            "LAYLA",
            "Hello! 👋 Main Layla hoon.\n"
            "Ye mera first app version hai."
        )

        return root

    def add_message(self, sender, text):
        label = Label(
            text=f"[b]{sender}:[/b] {text}",
            markup=True,
            size_hint_y=None,
            text_size=(None, None),
            halign="left"
        )

        label.bind(
            texture_size=lambda obj, value:
            setattr(obj, "height", value[1] + 20)
        )

        self.chat.add_widget(label)

    def ai_reply(self, message):
        msg = message.lower().strip()

        if msg in ["hi", "hello", "hey", "नमस्ते"]:
            return "Hello! 😊 Main Layla hoon."

        if "tumhara naam" in msg or "your name" in msg:
            return "Mera naam Layla hai. 🤖"

        if "kaise ho" in msg or "how are you" in msg:
            return "Main ready hoon! 😄"

        if "kya kar sakti ho" in msg:
            return (
                "Main tumhari personal AI assistant "
                "banne ke liye ready hoon."
            )

        return (
            "Main abhi basic version hoon. "
            "Agle versions mein memory aur AI brain add karenge. 🧠"
        )

    def send_message(self, instance):
        text = self.message.text.strip()

        if not text:
            return

        self.add_message("YOU", text)
        self.add_message("LAYLA", self.ai_reply(text))

        self.message.text = ""


if __name__ == "__main__":
    Layla().run()
