import speech_recognition as sr
from crewai_tools import tool
from audio.stt_provider.STT import STT
from Crewai.tools.base_tool import BaseTool

class recognize_speech(BaseTool):
    def __init__(self):
        super().__init__(
            name="Speech Recognition",
            description="Capture audio and convert to text."
        )
    def _run(self):
        """
        Capture audio and convert to text.
        :return:
        """
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source)
            try:
                stt=STT()
                stt.configure()
                text = stt.transcribe(audio)
                return f"Recognized Speech: {text}"
            except sr.UnknownValueError:
                return "Couldn't understand what you said."

