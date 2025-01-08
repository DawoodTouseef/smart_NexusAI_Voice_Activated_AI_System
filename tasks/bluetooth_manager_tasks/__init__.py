from Crewai import Task
from Crewai import Agent

class BluetoothManagerTasks:
    def __init__(self,agent:Agent):
        super().__init__()
        self.wifi_manager_agent=agent
    def scan_bluetooth_task(self):
        return Task(
            description= "Scan for available Bluetooth networks.",
            expected_output= "List of available Bluetooth networks.",
            agent=self.wifi_manager_agent
        )
    def connect_bluetooth_task(self):
        return Task(
            description="Connect to a specified Bluetooth network.",
            expected_output= "Bluetooth connection successful or error message.",
            agent=self.wifi_manager_agent
        )
    def disconnect_bluetooth_task(self):
        return Task(
            description="Disconnect from the current Bluetooth network.",
            expected_output= "Bluetooth disconnection successful or error message.",
            agent=self.wifi_manager_agent
        )
