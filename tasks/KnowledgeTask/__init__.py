from Crewai import Task
from Crewai import Agent



class KnowledgeTask:
    def __init__(self,agent:Agent):
        super().__init__()
        self.agent=agent

    def domain(self)->Task:
        return  Task(
            name="Domain-Specific Search",
            description="Provides accurate information specific to a field.",
            expected_output="Refined and precise search results.",
            agent=self.agent
        )
    def Live(self)->Task:
        return Task(
            name="Live Event Monitoring",
            description="Tracks and notifies about changes in ongoing events.",
            expected_output="Real-time notifications for dynamic events, like stock price changes,news etc..",
            agent=self.agent
        )

