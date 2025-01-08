from Crewai import Task,Agent

class Microsoft_365:
    def __init__(self,agent:Agent):
        super().__init__()
        self.microsoft_365_agent=agent

# Define tasks for Microsoft 365 applications
    def read_outlook_email_task(self):
        return Task(
    description="Read the latest email from Outlook.",
    expected_output="Latest Outlook email snippet.",
    agent=self.microsoft_365_agent,
        )

    def create_word_document_task(self):
        return Task(
    description="Create a new Word document with specified content.",
    expected_output="Word document created confirmation.",
    agent=self.microsoft_365_agent,
)

    def read_excel_file_task(self):
        return Task(
    description="Read the contents of an Excel file in OneDrive.",
    expected_output="Excel file content.",
    agent=self.microsoft_365_agent,
)

    def send_teams_message_task(self):
        return Task(
    description="Send a message to a Microsoft Teams channel.",
    expected_output="Teams message sent confirmation.",
    agent=self.microsoft_365_agent,
)
