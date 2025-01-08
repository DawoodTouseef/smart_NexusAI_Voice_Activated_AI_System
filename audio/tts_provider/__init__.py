from audio.tts_provider.TTS.Pyttsx3 import Pyttsx3
from audio.tts_provider.TTS.parlertts import ParlerTTS
from audio.tts_provider.TTS.main import BaseText2Speech

class TTS(BaseText2Speech):
    def __init__(self,provider:str="parler"):
        super().__init__()
        print(provider)
        self.primary_provider=ParlerTTS()
        self.secondary_provider=Pyttsx3()
        self.provider=provider
    def synthesize(self,text:str,provider:str=None):
        """

        :param text:
        :return:
        """
        try:
            from printing import printf
            printf(text,type="info")
            if self.provider=="parler" or (provider is not None and provider=="parler"):
                self.primary_provider.synthesize_text(text)
            else:
                self.secondary_provider.synthesize_text(text)
        except Exception as e:
            self.secondary_provider.synthesize_text(text)

