import os
import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from crewai_tools import tool
import requests

from env import (
    MICROSOFT_TENANT_ID,
    MICROSOFT_CLIENT_SECRET,
    MICROSOFT_CLIENT_ID,
)
from Crewai.tools.base_tool import BaseTool

OUTLOOK_CLIENT_ID =MICROSOFT_CLIENT_ID
OUTLOOK_CLIENT_SECRET = MICROSOFT_CLIENT_SECRET
OUTLOOK_TENANT_ID = MICROSOFT_TENANT_ID
OUTLOOK_SCOPE = "https://graph.microsoft.com/.default"
OUTLOOK_AUTH_URL = f"https://login.microsoftonline.com/{OUTLOOK_TENANT_ID}/oauth2/v2.0/token"
GRAPH_API_URL = "https://graph.microsoft.com/v1.0"
# Define SCOPES for Gmail
SCOPES = ['https://www.googleapis.com/auth/gmail.send', 'https://www.googleapis.com/auth/gmail.readonly']

def gmail_authenticate():
    """Authenticate and return the Gmail service instance."""
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    service = build('gmail', 'v1', credentials=creds)
    return service

class send_gmail(BaseTool):
    def __init__(self):
        super().__init__(
            name=f"{self.__class__.__name__}",
            description="Send an email using Gmail."
        )
    async def _run(to: str, subject: str, body: str):
        """Send an email using Gmail."""
        service = gmail_authenticate()
        message = MIMEMultipart()
        message['to'] = to
        message['subject'] = subject
        message.attach(MIMEText(body, 'plain'))
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        try:
            service.users().messages().send(userId="me", body={'raw': raw_message}).execute()
            return f"Email sent to {to}."
        except Exception as e:
            return f"Failed to send email: {e}"

class read_latest_gmail(BaseTool):
    def __init__(self):
        super().__init__(
            name="Read Email",
            description="Read the latest email in the Gmail inbox."
        )
    async def _run(self):
        """Read the latest email in the Gmail inbox."""
        service = gmail_authenticate()
        try:
            results = service.users().messages().list(userId='me', maxResults=1).execute()
            message = results.get('messages', [])[0]
            msg = service.users().messages().get(userId='me', id=message['id']).execute()
            snippet = msg['snippet']
            return f"Latest Email: {snippet}"
        except Exception as e:
            return f"Failed to read latest email: {e}"



def outlook_authenticate():
    """Authenticate with Microsoft Graph and retrieve access token."""
    response = requests.post(
        OUTLOOK_AUTH_URL,
        data={
            'client_id': OUTLOOK_CLIENT_ID,
            'client_secret': OUTLOOK_CLIENT_SECRET,
            'scope': OUTLOOK_SCOPE,
            'grant_type': 'client_credentials'
        }
    )
    response.raise_for_status()
    return response.json()['access_token']

class send_outlook_email(BaseTool):
    def __init__(self):
        super().__init__(
            name="Send Outlook Email",
            description="Send an email using Outlook."
        )
    async def _run(self,to: str, subject: str, body: str):
        """Send an email using Outlook."""
        token = outlook_authenticate()
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        email_data = {
            "message": {
                "subject": subject,
                "body": {
                    "contentType": "Text",
                    "content": body
                },
                "toRecipients": [
                    {
                        "emailAddress": {
                            "address": to
                        }
                    }
                ]
            }
        }
        response = requests.post(f"{GRAPH_API_URL}/me/sendMail", headers=headers, json=email_data)
        if response.status_code == 202:
            return f"Outlook email sent to {to}."
        else:
            return f"Failed to send Outlook email: {response.text}"

class read_latest_outlook_email(BaseTool):
    def __init__(self):
        super().__init__(
            name=f"{self.__class__.__name__}",
            description="Read the latest email in the Outlook inbox."
        )
    async def _run(self):
        """Read the latest email in the Outlook inbox."""
        token = outlook_authenticate()
        headers = {'Authorization': f'Bearer {token}'}
        response = requests.get(f"{GRAPH_API_URL}/me/messages?$top=1", headers=headers)
        if response.status_code == 200:
            email = response.json()['value'][0]
            return f"Latest Outlook Email: {email['subject']} - {email['bodyPreview']}"
        else:
            return f"Failed to read latest Outlook email: {response.text}"
