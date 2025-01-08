from Crewai import  Agent
from tools.email_manager_tool import send_gmail, read_latest_gmail, send_outlook_email, read_latest_outlook_email
from Crewai import LLM

class EmailManagerAgent:
    def __init__(self,llm:LLM):
        super().__init__()
        self.llm=llm

    def email_manager_agent(self):
        return Agent(
        role="Email Manager Agent",
        goal="Send and read emails using Gmail and Outlook.",
        backstory="Your an Email Manager Agent to send and read emails from Gmail and outlook",
        verbose=True,
        tools=[send_gmail(), read_latest_gmail(), send_outlook_email(), read_latest_outlook_email()],

        )