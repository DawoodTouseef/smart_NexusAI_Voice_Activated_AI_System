import pyttsx3
from crewai_tools import tool
from Crewai.tools.base_tool import BaseTool

class text_to_speech(BaseTool):
    def __init__(self):
        super().__init__(
            name="Speech Systesizer",
            description='Convert text to speech'
        )

    def _run(self,text: str):
        """
        Convert text to speech
        :param text:
        :return:
        """

        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()

