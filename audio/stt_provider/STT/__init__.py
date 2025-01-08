from typing import Dict, Union

from .main import BaseSpeech2Text

import speech_recognition as sr
from .Whispers import WhispersBase,WhispersLargeTurbo
from .nexusai import NexusSpeech


class STT(BaseSpeech2Text):
    def __init__(self,model:str=None,base_url:str=None,**kwargs):
        super().__init__()
        self.model=model
        self.base_url=base_url
        self.kwargs=kwargs


    def configure(self, **kwargs):
        """
        :param kwargs:
        :return:
        """
        self.model = kwargs.get("model")
        self.base_url = kwargs.get("base_url")
        self.kwargs = kwargs
    def transcribe(self, audio_data: Union[bytes,sr.AudioData], streaming: bool = False) -> Dict[str, Union[str, float]]:
        if isinstance(audio_data,bytes):
            audio_data=sr.AudioData(audio_data,sample_rate=16000,sample_width=2)
        try:
            self.configure()
            if  self.kwargs.get('organization') is not None:
                if self.kwargs.get("organization").lower()=="openai":
                    if self.model.lower()=="large-v3-turbo":
                        w=WhispersLargeTurbo()
                    else:
                        w=WhispersBase()
            else:
                from printing import printf
                printf("Using Nexus AI Speech Recognition")
                w=NexusSpeech()

            data=w.transcribe(audio_data, streaming)
            return data
        except Exception as e:
            return e