import os
from Crewai.tools.base_tool import BaseTool


class  list_files(BaseTool):
    def __init__(self):
        super().__init__(
            name="List Files",
            description= """List files and directories in the specified directory.""",
        )
    async def _run(self,directory: str = "."):
        """List files and directories in the specified directory."""
        try:
            files = os.listdir(directory)
            return f"Files in '{directory}': {', '.join(files)}" if files else f"No files found in '{directory}'."
        except Exception as e:
            return f"Failed to list files in '{directory}': {e}"
class create_directory(BaseTool):
    def __init__(self):
        super().__init__(
            name="Create Directory",
            description="Create a new directory.",
        )
    async def _run(self,directory: str):
        """Create a new directory."""
        try:
            os.makedirs(directory, exist_ok=True)
            return f"Directory '{directory}' created successfully."
        except Exception as e:
            return f"Failed to create directory '{directory}': {e}"

class delete_file(BaseTool):
    def __init__(self):
        super().__init__(
            name="Delete File",
            description="Delete a specified file."
            )
    async def _run(self,file_path: str):
        """Delete a specified file."""
        try:
            os.remove(file_path)
            return f"File '{file_path}' deleted successfully."
        except Exception as e:
            return f"Failed to delete file '{file_path}': {e}"

class rename_file(BaseTool):
    def __init__(self):
        super().__init__(
            name="Rename a File",
            description='Rename a specified file.'
        )
    async def _run(self,old_name: str, new_name: str):
        """Rename a specified file."""
        try:
            os.rename(old_name, new_name)
            return f"File '{old_name}' renamed to '{new_name}'."
        except Exception as e:
            return f"Failed to rename file '{old_name}': {e}"

class execute_command(BaseTool):
    def __init__(self):
        super().__init__(
            name="Execute Command",
            description="Execute a shell command and return its output."
        )
    async def _run(self,command: str):
        """Execute a shell command and return its output."""
        try:
            output = os.popen(command).read()
            return f"Command output:\n{output}" if output else "Command executed successfully with no output."
        except Exception as e:
            return f"Failed to execute command '{command}': {e}"

class get_env_variable(BaseTool):
    def __init__(self):
        super().__init__(
            name="get env variable",
            description="Retrieve the value of a specified environment variable."
        )
    async def _run(self,var_name: str):
        """Retrieve the value of a specified environment variable."""
        value = os.getenv(var_name)
        return f"Environment variable '{var_name}': {value}" if value else f"Environment variable '{var_name}' not found."

