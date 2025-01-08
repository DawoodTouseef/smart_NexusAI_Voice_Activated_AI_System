from Crewai import Agent
from llm import LLM

tools=[]
try:
    from tools.wifi_manager_tool import *

    tools.append(scan_wifi_networks())
    tools.append(connect_to_wifi())
    tools.append(disconnect_wifi())
except ModuleNotFoundError as e:
    pass
try:
    from tools.bluetooth_manager_tool import *

    tools.append(scan_bluetooth_devices)
    tools.append(connect_bluetooth_device)
    tools.append(disconnect_bluetooth_device)
except ModuleNotFoundError as e:
    pass

class connectivityAgent:
    def __init__(self,llm:LLM):
        super().__init__()
        self.llm=llm

    def connectivity_agent(self):
        return Agent(
            role= "Connectivity Manager",
            goal="Manage Wi-Fi and Bluetooth connections.",
            backstory="A dedicated agent for managing wireless connectivity and handling Bluetooth devices.",
            allow_delegation=False,
            tools=tools,
        )