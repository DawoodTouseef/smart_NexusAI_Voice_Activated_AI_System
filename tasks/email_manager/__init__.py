from Crewai import Task,Agent


class EmailManagerTasks:
    def __init__(self,agent:Agent):
        self.email_manager_agent=agent

    def send_gmail_task(self):
        return Task(
            description="Send an email via Gmail.",
            expected_output="Confirmation of email sent via Gmail.",
            agent=self.email_manager_agent
        )

    def read_gmail_task(self):
        return Task(
                description= "Read the latest email from Gmail.",
                expected_output="Snippet of the latest email in Gmail.",
                agent=self.email_manager_agent
        )

    def send_outlook_task(self):
        return Task(
            description="Send an email via Outlook.",
            expected_output="Confirmation of email sent via Outlook.",
            agent=self.email_manager_agent,
        )


    def read_outlook_task(self):
        return Task(
            description="Read the latest email from Outlook.",
            expected_output="Snippet of the latest email in Outlook.",
            agent=self.email_manager_agent,
        )

