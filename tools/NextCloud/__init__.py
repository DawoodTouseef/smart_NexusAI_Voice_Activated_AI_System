from pathlib import Path
from typing import Union, Optional, Dict, Any



from Crewai.tools.base_tool import BaseTool
import nextcloud_client
from env import (
    NEXTCLOUD_DOMAIN,
    NEXTCLOUD_PASSWORD,
    NEXTCLOUD_USERNAME,
)

class delete(BaseTool):
    def __init__(self):
        super().__init__(
            name="NextCloud Delete Directory",
            description="""
        A wrapper around the nextcloud.
        "Useful for when you need to delete the files to the nextCloud "
        :param path:The path of the Directory in the Cloud which user need to be delete
        :return:
        """
        )


    def _run(self, path: Union[str, Path]):
        """
        A wrapper around the nextcloud.
        "Useful for when you need to delete the files to the nextCloud "
        :param path:The path of the Directory in the Cloud which user need to be delete
        :return:
        """
        try:
            nc = nextcloud_client.Client(str(NEXTCLOUD_DOMAIN))
            nc.login(user_id=NEXTCLOUD_USERNAME, password=NEXTCLOUD_PASSWORD)
            nc.delete(path)
            return f"Deleteing the  File from the {path}  is successfully"
        except Exception as e:
            return f"Deleteing the  File from the {path}  is unsuccessfully"

class download_files(BaseTool):
    def __init__(self):
        super().__init__(
            name="NextCloud Download File",
            description="""
        A wrapper around the nextcloud.
        "Useful for when you need to download the files from the nextCloud "
        :param cloud_path:The path of the File in the Cloud which user need to be downloaded
        :param local_path:The path of the File in the local system where it would save the File.
        :return:
        """
        )

    def _run(self, cloud_path: Union[str, Path], local_path: Optional[Union[str, Path]] = None):
        """
        A wrapper around the nextcloud.
        "Useful for when you need to download the files from the nextCloud "
        :param cloud_path:The path of the File in the Cloud which user need to be downloaded
        :param local_path:The path of the File in the local system where it would save the File.
        :return:
        """
        try:
            nc = nextcloud_client.Client(str(NEXTCLOUD_DOMAIN))
            nc.login(user_id=NEXTCLOUD_USERNAME, password=NEXTCLOUD_PASSWORD)
            nc.get_file(remote_path=cloud_path,local_file=local_path)
            return f"Downloading the {cloud_path} File to the {local_path} is successfully"
        except Exception as e:
            return f"Downloading the {cloud_path} File to the {local_path} is unsuccessfully"

class get_files(BaseTool):
    def __init__(self):
        super().__init__(
            name= "NextCloud Get File",
            description="""
               A wrapper around the nextcloud.
               "Useful for when you need to get the list files from the nextCloud "
               :param path:The path of the File in the Cloud which user need to get the list of files
               :return:
        """
        )

    def _run(self, path: Optional[Union[str, Path]] = None) -> Union[str, Dict]:
        """
               A wrapper around the nextcloud.
               "Useful for when you need to get the list files from the nextCloud "
               :param path:The path of the File in the Cloud which user need to get the list of files
               :return:
        """
        try:
            nc = nextcloud_client.Client(str(NEXTCLOUD_DOMAIN))
            nc.login(user_id=NEXTCLOUD_USERNAME, password=NEXTCLOUD_PASSWORD)
            files=nc.list(path=path)
            return files
        except Exception as e:
            return f"No Directory or File Name founded in the NextCloud"

class mkdir(BaseTool):
    def __init__(self):
        super().__init__(
            name="Create Directory",
            description="""
                      A wrapper around the nextcloud.
                      "Useful for when you need to create a new Directory in the  nextCloud "
                      :param directory_name:name of the Directory and with the path of the Directory where the Directory shouldbe saved.
                      :return:

        """
        )
    def _run(self, directory_name: Union[str, Path]):
        """
                      A wrapper around the nextcloud.
                      "Useful for when you need to create a new Directory in the  nextCloud "
                      :param directory_name:name of the Directory and with the path of the Directory where the Directory shouldbe saved.
                      :return:

        """
        try:
            nc = nextcloud_client.Client(str(NEXTCLOUD_DOMAIN))
            nc.login(user_id=NEXTCLOUD_USERNAME, password=NEXTCLOUD_PASSWORD)
            nc.mkdir(path=directory_name)
            return "New Directory created succefully."
        except Exception as e:
            return "Directory already or existed."

class upload_directory(BaseTool):
    def __init__(self):
        super().__init__(
            name="NextCloud Uploader",
            description="""
        A wrapper around the nextcloud.
        "Useful for when you need to upload the directory to the nextCloud "
        "Input should be a Directory path of the local system and Directory path of the cloud where the Directory should upload."
        """
        )

    def _run(
        self,
        cloud_dir:str,
        local_dir:str
    ) -> Any:
        try:
            nc = nextcloud_client.Client(str(NEXTCLOUD_DOMAIN))
            nc.login(user_id=NEXTCLOUD_USERNAME, password=NEXTCLOUD_PASSWORD)
            nc.put_directory(target_path=cloud_dir,local_directory=local_dir)
            return f"Uploading the {local_dir} directory to the {cloud_dir} is successfully"
        except Exception as e:
            return f"Uploading the {local_dir} directory to the {cloud_dir} is unsuccessfully"

class upload_files(BaseTool):
    def __init__(self):
        super().__init__(
            name="NextCloud FileUploader",
            description="""
                A wrapper around the nextcloud.
                "Useful for when you need to upload the files to the nextCloud "
                "Input should be a Directory path of the local system and File path of the cloud where the File should upload."
                :param local_path:The path of the File which user need to upload to the cloud from the local System.
                :param cloud_path:The path of the File where user should upload the files from the local system.
                :return:
        """
        )


    def _run(self, local_path: Union[str, Path], cloud_path: Optional[Union[str, Path]] = None):
        """
                A wrapper around the nextcloud.
                "Useful for when you need to upload the files to the nextCloud "
                "Input should be a Directory path of the local system and File path of the cloud where the File should upload."
                :param local_path:The path of the File which user need to upload to the cloud from the local System.
                :param cloud_path:The path of the File where user should upload the files from the local system.
                :return:
        """
        try:
            nc = nextcloud_client.Client(str(NEXTCLOUD_DOMAIN))
            nc.login(user_id=NEXTCLOUD_USERNAME, password=NEXTCLOUD_PASSWORD)
            nc.put_file(remote_path=cloud_path,local_source_file=local_path)
            return f"Uploading the {local_path} File to the {cloud_path} is successfully"
        except Exception as e:
            return f"Uploading the {local_path} File to the {cloud_path} is unsuccessfully"