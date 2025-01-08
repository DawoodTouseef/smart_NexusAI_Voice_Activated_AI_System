import openwakeword
from env import (
    CACHE_DIR,
    POSITIVE_REFERENCE_CLIPS,
    NEGATIVE_REFERENCE_CLIPS
)
import os
from threading import Thread

def train_speaker_models():
    model_names=[]
    for i in model_names:
        def train_speaker():
            fil_name=str(i).split(".onnx")
            path = os.path.join(CACHE_DIR,"wakeword", f"{fil_name[0]}_verify.pkl")
            openwakeword.train_custom_verifier(
                positive_reference_clips=f"{POSITIVE_REFERENCE_CLIPS}",
                negative_reference_clips=f"{NEGATIVE_REFERENCE_CLIPS}",
                output_path=path,
                model_name=i
            )
        open_wake_words=Thread(target=train_speaker,args=())
        open_wake_words.start()
        open_wake_words.join()
