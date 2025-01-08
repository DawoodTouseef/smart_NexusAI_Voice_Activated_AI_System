import time
import asyncio
import logging
from abc import abstractmethod
from typing import Optional, Dict, Union, List
from collections import defaultdict
from langdetect import detect
from transformers import pipeline
import re
import inflect  # For converting numbers to words
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from ..cloud import Cloud
from translate import Translator


# Initialize lemmatizer, inflect engine, and stopwords list
lemmatizer = WordNetLemmatizer()
p = inflect.engine()
stop_words = set(stopwords.words('english'))

# Extended dictionary for common abbreviations and contractions
abbreviation_mapping = {
    "u": "you",
    "r": "are",
    "btw": "by the way",
    "lol": "laugh out loud",
    "omg": "oh my god",
    "idk": "I don't know",
    "brb": "be right back",
    "smh": "shaking my head",
    "tbh": "to be honest",
    "gr8": "great",
    "plz": "please",
    "thx": "thanks",
    "won't": "will not",
    "can't": "cannot",
    "n't": "not",
    "'ve": "have",
    "'ll": "will",
    "'re": "are",
    "'d": "would",
    "bcuz":"because",
    "don't":"do not",
}


class BaseText2Speech:
    def __init__(self, primary_provider=None, secondary_provider=None, cloud_storage:Cloud=None):
        self.primary_provider:BaseText2Speech = primary_provider
        self.secondary_provider:BaseText2Speech = secondary_provider
        self.cloud_storage = cloud_storage
        self.sessions = {}
        self.task_queue = asyncio.Queue()
        self.priorities = asyncio.PriorityQueue()
        self.user_statistics = defaultdict(
            lambda: {"total_requests": 0, "languages": defaultdict(int)})  # Track usage statistics
        self.language_support = ["en", "es", "fr", "de", "zh", "ar", "ru", "jp"]  # Enhanced language support
        self.custom_voice_models = {}  # Store custom voice models for users
        self.ssml_support = True  # Enable SSML input format
        self.sentiment_analysis_enabled = True  # Enable sentiment-based synthesis adjustments
        self.feedback_system = {}  # Store user feedback
        self.usage_limits = {"free": 100, "premium": 1000}  # Quota limits for different user tiers


    def start_session(self, session_id: str):
        self.sessions[session_id] = {"start_time": time.time()}
        logging.info(f"Session {session_id} started for user.")

    def end_session(self, session_id: str):
        if session_id in self.sessions:
            session_data = self.sessions.pop(session_id)
            duration = time.time() - session_data["start_time"]
            logging.info(f"Session {session_id} ended. Duration: {duration:.2f} seconds")
        else:
            logging.warning(f"Session {session_id} not found.")

    def get_session_data(self, session_id: str):
        return self.sessions.get(session_id, None)

    def process_text_in_batches(self,texts):
        for text in texts:
            yield text
    # --- New Feature: Real-time Text-to-Speech (Optimized Streaming) --- #
    async def real_time_stream(self, text: str, voice: Optional[str] = None):
        """
        Yield real-time synthesized audio chunks for immediate playback.
        Supports SSML and real-time buffering for a smooth experience.
        """
        for segment in self.process_text_in_batches(text):
            audio_data = await self.stream_synthesize(segment, voice)
            yield audio_data

    async def stream_synthesize(self, text: str, voice: Optional[str]):
        # Async streaming for real-time TTS
        audio_data = self.synthesize(text)
        await asyncio.sleep(0.1)  # Simulating a delay for real-time effect
        return audio_data

    def prioritize_request(self, user_id: str, text: str, voice: str, priority: int):
        self.priorities.put_nowait((priority, (text, voice, user_id)))

    # --- Enhanced Feature: Multi-language Support with Translation --- #
    def translate_text(self, text: str, target_language: str) -> str:
        """
        Translate the text into the target language before synthesizing speech.
        You could integrate with an external API like Google Translate or use an NLP model.
        """
        # Example logic to simulate translation
        logging.info(f"Translating text to {target_language}")
        translator = Translator(to_lang=target_language)
        return translator.translate(text)

    # --- Enhanced Feature: SSML with Extended Prosody Support --- #
    def synthesize_text(self, text: str,types="normal"):
        if types not in "normal":
            if self.ssml_support and "<speak>" in text:
                    logging.info("Processing SSML input with extended prosody control.")
            language = self.detect_language(text)
            text = self.normalize_text(text, language)
            logging.info(f"After applying normalization:{text}")
            text = self.translate_text(text, "en")
            logging.info(f"After translating {text} into {text}")
        self.synthesize(text)

    @abstractmethod
    def synthesize(self,text:str):
        pass


    # --- Enhanced Feature: Sentiment & Emotion-based Modulation --- #
    def apply_sentiment_analysis(self, text: str) -> Dict[str, float]:
        """
        Analyze the sentiment of the text to adjust speech tone, pitch, etc.
        """
        # Implement sentiment detection using a model like VADER or Hugging Face
        logging.info("Applying sentiment-based adjustments.")
        # Load the sentiment analysis pipeline
        sentiment_pipeline = pipeline("sentiment-analysis")
        result = sentiment_pipeline(text)[0]
        if result['label'] == 'POSITIVE':
            tone = "positive"
            pitch = 1.5  # Higher pitch for positive tone
        elif result['label'] == 'NEGATIVE':
            tone = "negative"
            pitch = 0.5  # Lower pitch for negative tone
        else:
            tone = "neutral"
            pitch = 1.0  # Base pitch for neutral tone

        # Output the desired format
        output = {"tone": tone, "pitch": pitch}
        return output

    # Function to expand abbreviations and contractions
    def expand_abbreviations(self,text):
        """Replace common abbreviations with their full forms."""
        tokens = word_tokenize(text)
        expanded_tokens = [abbreviation_mapping.get(token.lower(), token) for token in tokens]
        return ' '.join(expanded_tokens)

    # Convert numbers to words with contextual units
    def convert_numbers(self,text):
        """Convert numbers to words and handle currency and units."""

        def convert(match):
            number = match.group()
            return p.number_to_words(number)

        # Convert plain numbers
        text = re.sub(r'\b\d+\b', convert, text)

        # Convert specific units (currency, percentage, etc.)
        text = re.sub(r'\$(\d+)', lambda m: f"{p.number_to_words(m.group(1))} dollars", text)
        text = re.sub(r'(\d+)%', lambda m: f"{p.number_to_words(m.group(1))} percent", text)

        return text

    # Remove unnecessary punctuation but preserve meaningful ones
    def remove_punctuation(self,text):
        """Remove punctuation, keeping some specific ones like periods or question marks."""
        return re.sub(r'[^\w\s\.\?\!]', '', text)

    # Function to get WordNet POS tags for better lemmatization
    def get_wordnet_pos(self,treebank_tag):
        """Map NLTK POS tag to WordNet POS tag for better lemmatization."""
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def normalize_text(self, text: str, language: str) -> str:
        # Implement text normalization, handling abbreviations, numbers, etc.
        """Full normalization pipeline with more complex handling."""
        # Step 1: Lowercasing
        text = text.lower()

        # Step 2: Expand abbreviations and contractions
        text = self.expand_abbreviations(text)

        # Step 3: Convert numbers to words with context
        text = self.convert_numbers(text)

        # Step 4: Remove unnecessary punctuation
        text = self.remove_punctuation(text)

        # Step 5: Tokenization and POS tagging
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)

        # Step 6: Lemmatize based on POS tagging for better accuracy
        lemmatized_tokens = [lemmatizer.lemmatize(token, self.get_wordnet_pos(tag)) for token, tag in pos_tags]

        # Step 7: Remove stopwords if desired
        lemmatized_tokens = [token for token in lemmatized_tokens if token not in stop_words]

        # Remove extra spaces and return normalized text
        return ' '.join(lemmatized_tokens).strip()

    def detect_language(self, text: str) -> str:
        # Language detection logic (e.g., using langdetect)
        return detect(text)


    def store_audio(self):
        if self.cloud_storage:
            file_name = f"{time.time()}.mp3"
            self.cloud_storage.upload(file_name,"Jarvis/audio/transcription")
            logging.info(f"Audio stored in cloud storage: {file_name}")

    def fallback_to_secondary_provider(self, text: str, voice: Optional[str] = None):
        if self.secondary_provider:
            logging.warning("Primary provider failed, falling back to secondary provider.")
            return self.secondary_provider.synthesize(text)
        else:
            raise Exception("Both primary and secondary providers failed.")
