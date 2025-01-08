from Crewai import Agent
from Crewai import LLM
from tools.recognize_speech import recognize_speech
from tools.text_to_speech import text_to_speech

class conversationAgent:
    def __init__(self,llm:LLM):
        super().__init__()
        self.llm=llm

    def conversation_agent(self):
        return Agent(
                role='Conversational AI Agent',
                goal='Engage in conversations, understanding context and responding dynamically.',
                backstory="Handles general queries, small talk, and understanding context to provide intelligent responses.",
                verbose=True,
                tools=[recognize_speech(), text_to_speech()]
)
