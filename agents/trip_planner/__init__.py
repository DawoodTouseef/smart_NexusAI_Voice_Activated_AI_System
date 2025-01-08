from Crewai import Agent
from Crewai import   LLM
from tools.Search_Engine import BrowserTools
from tools.Search_Engine.bing import Bing
from tools.Search_Engine.brave import BraveSearchTool
from tools.calculator import CalculatorTool
from env import BRAVE_API_KEY,BING_SUBSCRIPTION_KEY



class TripAgents:
    def __init__(self,llm:LLM):
        super().__init__()
        self.llm=llm
    def city_selection_agent(self):
        tools = []
        if BING_SUBSCRIPTION_KEY:
              tools.append(Bing())
        if BRAVE_API_KEY:
              tools.append(BraveSearchTool())
        tools.append(BrowserTools())
        return Agent(
            role='City Selection Expert',
            goal='Select the best city based on weather, season, and prices',
            backstory='An expert in analyzing travel data to pick ideal destinations',
            tools=tools,
            llm=self.llm
            )

    def local_expert(self):
        tools = []
        if BING_SUBSCRIPTION_KEY:
              tools.append(Bing())
        if BRAVE_API_KEY:
              tools.append(BraveSearchTool())
        tools.append(BrowserTools())
        return Agent(
            role='Local Expert at this city',
            goal='Provide the BEST insights about the selected city',
            backstory="""A knowledgeable local guide with extensive information
            about the city, it's attractions and customs""",
            tools=tools,
            llm=self.llm
        )

    def travel_concierge(self):
        tools = []
        if BING_SUBSCRIPTION_KEY:
              tools.append(Bing())
        if BRAVE_API_KEY:
              tools.append(BraveSearchTool())
        tools.append(BrowserTools())
        tools.append(CalculatorTool())
        return Agent(
            role='Amazing Travel Concierge',
            goal="""Create the most amazing travel itineraries with budget and 
            packing suggestions for the city""",
            backstory="""Specialist in travel planning and logistics with 
            decades of experience""",
            tools=tools,
            llm=self.llm
        )

