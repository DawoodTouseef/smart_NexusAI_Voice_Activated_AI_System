import nextcloud_client
from pathlib import Path

from  .main import Cloud
try:
    from env import NEXTCLOUD_DOMAIN
except ImportError as e:
    pass
from printing import printf

class NextCloud(Cloud):
    def __init__(self):
        super().__init__()
        try:
            self.nc=nextcloud_client.Client(NEXTCLOUD_DOMAIN)
        except Exception  as e:
            self.nc=None
    def login(self,username:str,password:str):
        if self.nc is not None:
            self.nc.login(user_id=username,password=password)
            printf("Login Successfully")
    def check_files(self):
        try:
            files=self.nc.list("Jarvis")
        except Exception as e:
            self.nc.mkdir("Jarvis")
        try:
            files = self.nc.list("Jarvis/audio")
        except Exception as e:
            self.nc.mkdir("Jarvis/audio")
        try:
            files = self.nc.list("Jarvis/audio/transcription")
        except Exception as e:
            self.nc.mkdir("Jarvis/audio/transcription")

    def upload(self,filename:str,remotefilename:str):
        file=Path(filename)
        if self.nc is not None:
            self.nc.put_file(remote_path=f"{remotefilename}/{file.name}",local_source_file=filename)
            printf(f"File Name:{file.name}")

            printf(f"{file.name} uploaded successfully")

