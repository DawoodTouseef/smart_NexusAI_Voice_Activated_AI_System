from Crewai import Task,Agent

class OSManager:
    def __init__(self,agent:Agent):
        super().__init__()
        self.system_manager_agent=agent
    # Define system-specific tasks
    def list_files_task(self)->Task:
        return Task(
        description="List files and directories in the specified directory.",
        expected_output="Directory listing.",
        agent=self.system_manager_agent,
    )

    def create_directory_task(self)->Task:
        return Task(
        description="Create a new directory.",
        expected_output="Confirmation of directory creation.",
        agent=self.system_manager_agent,
    )

    def delete_file_task(self):
        return Task(
        description="Delete a specified file.",
        expected_output="File deletion confirmation.",
        agent=self.system_manager_agent,
    )
    def rename_file_task(self)->Task: 
        return Task(
        description="Rename a specified file.",
        expected_output="File renaming confirmation.",
        agent=self.system_manager_agent,
    )

    def execute_command_task(self):
        return Task(
        description="Execute a specified shell command and return the output.",
        expected_output="Shell command output.",
        agent=self.system_manager_agent,
    )

    def get_env_variable_task(self)->Task:
        return Task(
        description="Retrieve the value of a specified environment variable.",
        expected_output="Environment variable value.",
        agent=self.system_manager_agent,
    )