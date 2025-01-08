import os.path

from autogenstudio import WorkflowManager
from langchain.tools import Tool
from tensorboard.compat.tensorflow_stub.errors import exception_type_from_error_code

from env import AGENTS
import yaml
import autogen

def agenthook(workflow: str):
    """Load the Agent from the Workflow """
    workflow_manager = WorkflowManager(workflow=workflow)
    return workflow_manager
try:
# Load agents configuration from YAML file
    agents = yaml.load(open(os.path.join(str(AGENTS),"agents.yml")), Loader=yaml.FullLoader)
except FileNotFoundError as e:
    agents=None
def get_agent_studio_tool():
    tools=[]
    if agents is not None:
        tools = [
                Tool(
                    name=agent['name'],
                    func=lambda workflow=agent['workflow']: agenthook(workflow).run,
                    description=agent['description']
                ) for agent in agents
        ]

    return tools


