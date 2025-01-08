from Crewai import Task


class WifiManagerTasks:
    def __init__(self,agent):
        super().__init__()
        self.wifi_manager_agent=agent
    def scan_wifi_task(self):
        return Task(
            description= "Scan for available Wi-Fi networks.",
            expected_output= "List of available Wi-Fi networks.",
            agent=self.wifi_manager_agent
        )
    def connect_wifi_task(self):
        return Task(
            description="Connect to a specified Wi-Fi network.",
            expected_output= "Wi-Fi connection successful or error message.",
            agent=self.wifi_manager_agent
        )
    def disconnect_wifi_task(self):
        return Task(
            description="Disconnect from the current Wi-Fi network.",
            expected_output= "Wi-Fi disconnection successful or error message.",
            agent=self.wifi_manager_agent
        )
