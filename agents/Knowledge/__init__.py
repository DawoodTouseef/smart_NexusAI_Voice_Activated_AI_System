from Crewai import Agent
from llm import LLM
from tools.calculator import CalculatorTool
from tools.arxiv import ArxivQueryRun
from env import (
    BING_SUBSCRIPTION_KEY,
    BRAVE_API_KEY
)
from tools.Search_Engine.brave import BraveSearchTool
from tools.Search_Engine.bing import Bing

class KnowledgeAgent:
    def __init__(self,llm:LLM):
        super().__init__()
        self.llm=llm

    def agent(self)->Agent:
        tools = [
            CalculatorTool(),
            ArxivQueryRun(),
        ]
        if BING_SUBSCRIPTION_KEY is not None:
            web=Bing()
            tools.append(web)
        elif BRAVE_API_KEY is not None:
            web=BraveSearchTool()
            tools.append(web)
        return Agent(
            role="Internet Search Agent",
            goal="Enhanced real-time search with personalized filtering and advanced query refinement.",
            backstory="real-time search with personalized filtering and advanced query refinement",
            llm=self.llm,
            tools=tools

        )

