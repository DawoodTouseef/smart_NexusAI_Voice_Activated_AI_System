import os.path

from audio.stt_provider.STT.main import BaseSpeech2Text
import requests
from env import (
    NEXUSAI_API_KEY,
    NEXUSAI_API_BASE_URL,
    CACHE_DIR,
)

from  typing import Dict,Union
import speech_recognition as sr


class NexusSpeech(BaseSpeech2Text):
    def __init__(self):
        super().__init__()

    def configure(self, **kwargs):
        self.api_key=NEXUSAI_API_KEY
        self.base_url=NEXUSAI_API_BASE_URL

    def transcribe_audio(self, token: str, file_path: str):
        """Transcribe audio by sending it to the API."""
        # Prepare the file for upload
        file_name = os.path.basename(file_path)

        with open(file_path, 'rb') as file:
            # Remove 'content_type' and 'filename' from the files dictionary
            files = {
                'file': (file_name, file, 'audio/wav'),  # Provide filename and content type
            }
            # Set the headers
            headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {token}',

            }
            # Build the URL
            url = f"{str(self.base_url).rstrip('/')}/audio/api/v1/transcriptions"

            # Send the POST request to the API
            response = requests.post(url, headers=headers, files=files)

            # Check for errors in the response
            if response.status_code != 200:
                error = response.json().get('detail', 'An error occurred')
                return f"Error: {error}"

            return response.json()

    def transcribe(self, audio_data: sr.AudioData, streaming: bool = False) -> Dict[str, Union[str, float]]:
        """

        :param audio_data:
        :param streaming:
        :return:
        """
        # Convert audio_data (bytes) to AudioData object
        self.configure()
        try:
            audio_dir=os.path.join(CACHE_DIR,"audio","transcription")
            if not os.path.exists(audio_dir):
                os.makedirs(audio_dir,exist_ok=True)
            i = len([name for name in os.listdir(audio_dir) if os.path.isfile(os.path.join(audio_dir, name))])
            filename = os.path.join(audio_dir, f"audio-{i + 1}.wav")
            with open(filename,"wb") as f:
                f.write(audio_data.get_wav_data())
            data=self.transcribe_audio(self.api_key,filename)
            return data['text']
        except sr.UnknownValueError:
            return "Could not understand the audio"
        except sr.RequestError as e:
            return f"Could not request results from Whisper Speech Recognition service; {e}"


if __name__=="__main__":
    n=NexusSpeech()
    r=sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        print("Listening...")
        audio=r.listen(source)
        print("recognizing...")
        text=n.transcribe(audio)
        print(text)