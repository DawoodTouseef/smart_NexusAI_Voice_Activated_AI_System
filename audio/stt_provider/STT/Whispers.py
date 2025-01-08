from .main import BaseSpeech2Text
from typing import Dict,Union

import  speech_recognition as sr

class WhispersBase(BaseSpeech2Text):
    def __init__(self):
        super().__init__()
    def configure(self, **kwargs):
        self.api=kwargs.get("api")

    def transcribe(self, audio_data: sr.AudioData, streaming: bool = False) -> Dict[str, Union[str, float]]:
        # Convert audio_data (bytes) to AudioData object
        audio = audio_data

        # Try using Google Web Speech API for transcription
        try:
            text = sr.Recognizer().recognize_whisper(audio,show_dict=True)

            # Returning result with a dummy confidence score as the SpeechRecognition package doesn't provide it
            return text
        except sr.UnknownValueError:
            return "Could not understand the audio"
        except sr.RequestError as e:
            return f"Could not request results from Whisper Speech Recognition service; {e}"

class WhispersLargeTurbo(BaseSpeech2Text):
    def __init__(self):
        super().__init__()
    def configure(self, **kwargs):
        self.api=kwargs.get("api")

    def transcribe(self, audio_data: sr.AudioData, streaming: bool = False) -> Dict[str, Union[str, float]]:
        # Convert audio_data (bytes) to AudioData object
        audio = audio_data

        # Try using Google Web Speech API for transcription
        try:

                # Use Google's free API without API key
            generate_kwargs = {
                    "compression_ratio_threshold": 1.35,  # zlib compression ratio threshold (in token space)
                    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                    "logprob_threshold": -1.0,
                    "no_speech_threshold": 0.6,
                }
            text = sr.Recognizer().recognize_whisper(audio,model="large-v3-turbo",**generate_kwargs,show_dict=True)

            # Returning result with a dummy confidence score as the SpeechRecognition package doesn't provide it
            return text
        except sr.UnknownValueError:
            return "Could not understand the audio"
        except sr.RequestError as e:
            return f"Could not request results from Google Speech Recognition service; {e}"