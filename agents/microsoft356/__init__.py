from Crewai import Agent
from Crewai import LLM
from  tools.microsoft365 import *

class Microsoft365Agent:
    def __init__(self,llm:LLM):
        super().__init__()
        self.llm=llm
    # Define Microsoft 365 Agent
    def microsoft_365_agent(self):
        return Agent(
        role="Microsoft 365 Agent",
        goal="Handle tasks related to Microsoft 365 applications, including Outlook, Word, Excel, and Teams.",
        backstory="An office assistant capable of managing emails, documents, spreadsheets, and team communications.",
        tools=[read_outlook_email(), create_word_document(), read_excel_file(), send_teams_message()],
        allow_delegation=False,
        llm=self.llm,
)