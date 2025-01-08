from Crewai import Task,Agent

class HomeAssistantTask:
    def __init__(self,agent:Agent):
        super().__init__()
        self.agent=agent

    def check_status_task(self)->Task:
        return Task(
            description="Check the status of all devices in the house.",
            expected_output="A list of devices and their current statuses.",
            agent=self.agent,
            human_input=True,
        )
    def control_task(self)->Task:
        return Task(
            description="Control devices in the house using a command (e.g., 'Turn on the living room light').",
            expected_output="Acknowledgment of the action performed.",
            agent=self.agent,
            human_input=True
        )

