import pyttsx3
from audio.tts_provider.TTS.main import BaseText2Speech

class Pyttsx3(BaseText2Speech):
    def __init__(self):
        super().__init__()
        self.engine=pyttsx3.init()

    def synthesize(self,text):
        self.engine.setProperty('rate', 125)  # setting up new voice rate
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice',voices[0].id)
        self.engine.say(text)
        self.engine.runAndWait()

