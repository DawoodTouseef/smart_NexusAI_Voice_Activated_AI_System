import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
from audio.tts_provider.TTS.main import BaseText2Speech
import sounddevice as sd
import git
from printing import printf
from env import JARVIS_DIR
import os

def download_parlertts():
    """
    :return:
    """
    path=os.path.join(JARVIS_DIR,"config","model","parler-tts")
    if not os.path.exists(path):
        printf("Downloading the model..",type="warn")
        git.Repo.clone_from("https://huggingface.co/parler-tts/parler-tts-mini-multilingual",
                            to_path=f"{path}")
        printf("Model downloaded successfully")
    else:
        printf("Model Already downloaded.",type="warn")

class ParlerTTS(BaseText2Speech):
    def __init__(self):
        super().__init__()

    def synthesize(self,text):
        """
        The Function to synthesize
        :param text:
        :return:
        """
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        local_path=os.path.join(JARVIS_DIR,"config","model","parler-tts","parler-tts-mini-multilingual")
        model = ParlerTTSForConditionalGeneration.from_pretrained(local_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(local_path)
        model.half()
        description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

        description = "A deep, robotic male voice, calm, precise, and slightly monotone. The speech is clear, authoritative, and measured, delivered with a controlled pace and even tone, resembling a highly intelligent AI assistant"
        input_ids = description_tokenizer(description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

        generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        audio_arr = generation.cpu().numpy().squeeze()
        sd.play(audio_arr, model.config.sampling_rate)
        sd.wait()

