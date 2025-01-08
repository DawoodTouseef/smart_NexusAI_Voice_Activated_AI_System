from Crewai import Agent, Task
from textwrap import dedent


# Task Definitions for NextCloud operations
class NextCloudTask:
    def __init__(self,agent:Agent):
            super().__init__()
            self.agent=agent
            
    def local_file_and_dir(self):
        return Task(
            description=dedent("""
            Analyze the local system and return all the files and directories present.
            """),
            agent=self.agent,
            expected_output="A list of directories and files from the local system."
        )

    def upload_file(self):
        return Task(
            description=dedent("""
            Upload a file from the local system to the NextCloud instance.
            """),
            agent=self.agent,
            expected_output="The file is uploaded to NextCloud successfully."
        )

    def upload_directory(self):
        return Task(
            description=dedent("""
            Upload a directory from the local system to the NextCloud instance.
            """),
            agent=self.agent,
            expected_output="The directory is uploaded to NextCloud successfully."
        )

    def download_file(self):
        return Task(
            description=dedent("""
            Download a file from the NextCloud instance to the local system.
            """),
            agent=self.agent,
            expected_output="The file is downloaded to the local system."
        )

    def delete_directory(self):
        return Task(
            description=dedent("""
            Delete a directory from the NextCloud instance.
            """),
            agent=self.agent,
            expected_output="The directory is deleted from the cloud."
        )

    def get_files_and_directories(self):
        return Task(
            description=dedent("""
            Fetch all files and directories from the NextCloud instance.
            """),
            agent=self.agent,
            expected_output="A list of all files and directories in the NextCloud."
        )

    def create_directory(self):
        return Task(
            description=dedent("""
            Create a new directory in the NextCloud instance.
            """),
            agent=self.agent,
            expected_output="A new directory is created in NextCloud."
        )


