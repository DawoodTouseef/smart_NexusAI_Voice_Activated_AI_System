from Crewai import Agent
from llm import LLM
from tools.NextCloud import *


from textwrap import dedent

import os
from typing import Any, Optional, Type

from pydantic import BaseModel, Field

from Crewai.tools.base_tool import BaseTool


class FixedDirectoryReadToolSchema(BaseModel):
    """Input for DirectoryReadTool."""

    pass


class DirectoryReadToolSchema(FixedDirectoryReadToolSchema):
    """Input for DirectoryReadTool."""

    directory: str = Field(..., description="Mandatory directory to list content")


class DirectoryReadTool(BaseTool):
    name: str = "List files in directory"
    description: str = (
        "A tool that can be used to recursively list a directory's content."
    )
    args_schema: Type[BaseModel] = DirectoryReadToolSchema
    directory: Optional[str] = None

    def __init__(self, directory: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if directory is not None:
            self.directory = directory
            self.description = f"A tool that can be used to list {directory}'s content."
            self.args_schema = FixedDirectoryReadToolSchema
            self._generate_description()

    def _run(
        self,
        **kwargs: Any,
    ) -> Any:
        directory = kwargs.get("directory", self.directory)
        if directory[-1] == "/":
            directory = directory[:-1]
        files_list = [
            f"{directory}/{(os.path.join(root, filename).replace(directory, '').lstrip(os.path.sep))}"
            for root, dirs, files in os.walk(directory)
            for filename in files
        ]
        files = "\n- ".join(files_list)
        return f"File paths: \n-{files}"

class NextCloudAgent:
    def __init__(self,llm:LLM):
        super().__init__()
        self.llm=llm
    def run(self)->Agent:
        return  Agent(
    role='Cloud File Manager',
    goal='Manage files on NextCloud including uploading, downloading, and managing directories and files.',
    backstory=dedent("""
    As a Cloud File Manager,enhance file management on NextCloud simplifying tasks like uploading,downloading, and managing directories and files.
    Efficiently  manage directories and ensure secure access to sensitive information.
    """),
    tools=[
            mkdir(),
           delete(),
           get_files(),
           upload_files(),
           download_files(),
           upload_directory(),
           DirectoryReadTool()
    ],
    llm=self.llm,
    allow_delegation=True,
    )