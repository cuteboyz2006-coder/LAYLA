class TextData:
    def __init__(self):
        self.data = []

    def add(self, text):
        text = text.strip()

        if text:
            self.data.append(text)

    def get_all(self):
        return self.data
