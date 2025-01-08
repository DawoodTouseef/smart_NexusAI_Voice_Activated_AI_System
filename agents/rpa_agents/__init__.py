from Crewai import Agent
from tools.rpa_tool import *
from Crewai import LLM

# Define RPA Agent
class RPAAgent:
    def __init__(self,llm:LLM):
        super().__init__()
        self.llm=llm
    def rpa_agent(self):
        return Agent(
    role="RPA Agent",
    goal="Automate web interactions and GUI tasks using RPA.",
    backstory="A digital assistant capable of performing repetitive tasks and web automation.",
    tools=[open_website(), click_element(), type_text(), take_screenshot(), get_clipboard(), set_clipboard()],
    allow_delegation=False,
    llm=self.llm,
    )