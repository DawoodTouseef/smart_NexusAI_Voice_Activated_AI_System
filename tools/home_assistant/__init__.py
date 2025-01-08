from Crewai.tools.base_tool import BaseTool
import requests
from env import HA_BASE_URL,HA_API_KEY


class HomeAssistant(BaseTool):
    """
        Comprehensive Tool for Home Assistant device control and monitoring using the REST API.
        """

    def __init__(self):
        super().__init__(
            name="EnhancedHomeAssistantTool",
            description="Control and monitor Home Assistant devices and automations."
        )
        base_url=HA_BASE_URL
        token=HA_API_KEY
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

    def trigger_service(self, domain, service, payload):
        """Trigger a service in Home Assistant."""
        url = f"{self.base_url}/api/services/{domain}/{service}"
        response = requests.post(url, headers=self.headers, json=payload)
        return response.json() if response.status_code in [200, 201] else {"error": response.text}

    def get_entity_state(self, entity_id):
        """Fetch the state of a specific entity."""
        url = f"{self.base_url}/api/states/{entity_id}"
        response = requests.get(url, headers=self.headers)
        return response.json() if response.status_code == 200 else {"error": response.text}

    def _run(self, command):
        """
        Execute a parsed command.
        Supported actions:
        - "turn_on <entity_id>"
        - "turn_off <entity_id>"
        - "status <entity_id>"
        """
        if command.startswith("turn_on"):
            entity_id = command.split(" ")[1]
            return self.trigger_service(entity_id.split(".")[0], "turn_on", {"entity_id": entity_id})
        elif command.startswith("turn_off"):
            entity_id = command.split(" ")[1]
            return self.trigger_service(entity_id.split(".")[0], "turn_off", {"entity_id": entity_id})
        elif command.startswith("status"):
            entity_id = command.split(" ")[1]
            return self.get_entity_state(entity_id)
        else:
            return {"error": "Unsupported command."}

