from Crewai import Task,Agent

class Conversation_Task:
    def __init__(self,agent:Agent):
        super().__init__()
        self.agent=agent

    # Define Tasks
    def conversation_task(self):
        return Task(
        description="Engage in conversation and respond based on context.",
        expected_output="Accurate and context-aware responses.",
        agent=self.agent
        )
