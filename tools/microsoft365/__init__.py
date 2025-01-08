import requests
from crewai_tools import tool
from env import (
    MICROSOFT_TENANT_ID,
    MICROSOFT_CLIENT_ID,
    MICROSOFT_CLIENT_SECRET
)
from Crewai.tools.base_tool import BaseTool

# Constants for Microsoft API credentials
MICROSOFT_AUTH_URL = f"https://login.microsoftonline.com/{MICROSOFT_TENANT_ID}/oauth2/v2.0/token"
MICROSOFT_SCOPE = "https://graph.microsoft.com/.default"
GRAPH_API_URL = "https://graph.microsoft.com/v1.0"

def get_access_token():
    """Authenticate with Microsoft Graph and retrieve an access token."""
    response = requests.post(
        MICROSOFT_AUTH_URL,
        data={
            'client_id': MICROSOFT_CLIENT_ID,
            'client_secret': MICROSOFT_CLIENT_SECRET,
            'scope': MICROSOFT_SCOPE,
            'grant_type': 'client_credentials'
        }
    )
    response.raise_for_status()
    return response.json().get('access_token')


class  read_outlook_email(BaseTool):
    def __init__(self):
        super().__init__(
            name="Read Outlook Email",
            description="Read the latest email from Outlook."
        )
    async def _run(self):
        """Read the latest email from Outlook."""
        token = get_access_token()
        headers = {'Authorization': f'Bearer {token}'}
        response = requests.get(f"{GRAPH_API_URL}/me/messages?$top=1", headers=headers)
        if response.status_code == 200:
            email = response.json()['value'][0]
            return f"Latest Outlook Email: {email['subject']} - {email['bodyPreview']}"
        else:
            return f"Failed to read Outlook email: {response.text}"

class create_word_document(BaseTool):
    def __init__(self):
        super().__init__(name="Create a Word Document",description="Create a Word document in OneDrive with specified content.")
    async def _run(self,file_name: str, content: str):
        """Create a Word document in OneDrive with specified content."""
        token = get_access_token()
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        file_data = {
            "name": file_name,
            "content": content
        }
        response = requests.put(f"{GRAPH_API_URL}/me/drive/root:/{file_name}:/content", headers=headers, data=file_data)
        if response.status_code == 201:
            return f"Word document '{file_name}' created successfully."
        else:
            return f"Failed to create Word document: {response.text}"

class read_excel_file(BaseTool):
    def __init__(self):
        super().__init__(
            name=f"{self.__class__.__name__}",
            description="Read the content of an Excel file in OneDrive."
        )
    async def _run(self,file_name: str):
        """Read the content of an Excel file in OneDrive."""
        token = get_access_token()
        headers = {'Authorization': f'Bearer {token}'}
        response = requests.get(f"{GRAPH_API_URL}/me/drive/root:/{file_name}:/content", headers=headers)
        if response.status_code == 200:
            return f"Content of Excel file '{file_name}':\n{response.content.decode()}"
        else:
            return f"Failed to read Excel file: {response.text}"

class send_teams_message(BaseTool):
    def __init__(self):
        super().__init__(
            name="sen teams Message",
            description="""Send a message to a specified Microsoft Teams channel."""
        )
    async def _run(self,team_id: str, channel_id: str, message: str):
        """Send a message to a specified Microsoft Teams channel."""
        token = get_access_token()
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        message_data = {
            "body": {
                "content": message
            }
        }
        response = requests.post(f"{GRAPH_API_URL}/teams/{team_id}/channels/{channel_id}/messages", headers=headers, json=message_data)
        if response.status_code == 201:
            return "Message sent successfully to Teams channel."
        else:
            return f"Failed to send message to Teams: {response.text}"
