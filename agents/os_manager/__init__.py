from Crewai import Agent

from tools.os_manager import *
from llm import LLM


class OSManagerCrew:
    def __init__(self,llm:LLM):
        super().__init__()
        self.llm=llm
    def system_manager_agent(self)->Agent:
        return Agent(
            role="System Manager Agent",
            goal="Handle system-level tasks such as file operations, executing commands, and managing environment variables.",
            verbose=True,
            tools=[list_files(), create_directory(), delete_file(), rename_file(), execute_command(), get_env_variable()],
            allow_delegation=False,
            llm=self.llm,
            backstory="You can Handle system-level tasks such as file operations, executing commands, and managing environment variables."
        )