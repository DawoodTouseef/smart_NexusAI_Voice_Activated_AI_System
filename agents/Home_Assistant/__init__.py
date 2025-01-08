from Crewai import Agent,LLM
from tools.home_assistant import HomeAssistant
from requests import get


url = "https://fa61-202-12-81-250.ngrok-free.app/api/services"
headers = {
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiIxMmNmZjUwMzk2N2I0MjRkOTk2ZGZlM2ZkNzlhM2E5NiIsImlhdCI6MTczMjk2OTU5MSwiZXhwIjoyMDQ4MzI5NTkxfQ.vcMM2YYvNBDdTQdv1vNkL5Mz-ZZ1yxQSxzEA_pk2wLg",
    "content-type": "application/json",
}
response = get(url, headers=headers).json()
services=[]
devices={}
for i in response:
    for j in i['services'].keys():
        if j=="turn_on":
            state="on"
            devices.update({f"{i['domain']}.{j} '{i['services'][j]['description']}'":state})
        elif j=="turn_off":
            state="off"
            devices.update({f"{i['domain']}.{j} {i['services'][j]['description']}": state})
        services.append(f"{i['domain']}.{j}")
devices_str=""
for i in devices.keys():
    devices_str+=f"{i}={devices[i]}\n"
class HomeAssistantAgent:
    def __init__(self,llm:LLM):
        super().__init__()
        self.llm=llm

    def assistant_agent(self)->Agent:
        return Agent(
            role="Home Assistant",
            goal="Assist with smart home control using natural language.",
            backstory=(
                "You are 'Al', a helpful AI Assistant that controls the devices in a house. "
                "Complete tasks as instructed with the information provided only."
                f"Services:{",".join(services)}"
                "Devices:"
                f"{devices_str}"
            ),
            tools=[HomeAssistant()],
            verbose=True
        )
