from abc import ABC, abstractmethod
from typing import Union, Optional, Dict
from pathlib import Path


class Cloud(ABC):
    def __init__(self):
        """
        Initialize the Cloud object.
        """
        super().__init__()

    @abstractmethod
    def configure(self, config: Optional[Dict] = None) -> None:
        """
        Configure the cloud provider with the given settings.

        :param config: A dictionary containing configuration details.
        :return: None
        """
        pass

    @abstractmethod
    def get_files(self, path: Optional[Union[str, Path]] = None) -> Union[str, Dict]:
        """
        Fetch a list of files or directories from the cloud.

        :param path: The cloud directory path to fetch the list from. If not provided, fetch from root.
        :return: A string representing the file or a dictionary of files/directories.
        """
        pass

    def get_provider_name(self) -> str:
        """
        Returns the name of the cloud provider.

        :return: Cloud provider's name.
        """
        return self.__class__.__name__

    @abstractmethod
    def upload_files(self, local_path: Union[str, Path], cloud_path: Optional[Union[str, Path]] = None) -> None:
        """
        Upload files from the local system to the cloud.

        :param local_path: Path of the local file to upload.
        :param cloud_path: Cloud destination path. If not provided, upload to the root directory.
        :return: None
        """
        pass

    @abstractmethod
    def mkdir(self, directory_name: Union[str, Path]) -> None:
        """
        Create a new directory in the cloud.

        :param directory_name: Name of the directory to create.
        :return: None
        """
        pass

    @abstractmethod
    def delete(self, path: Union[str, Path]) -> None:
        """
        Delete a directory from the cloud.

        :param path: Path of the directory to delete.
        :return: None
        """
        pass

    @abstractmethod
    def upload_directory(self, local_dir: Union[str, Path], cloud_dir: Optional[Union[str, Path]] = None) -> None:
        """
        Upload an entire directory from the local system to the cloud.

        :param local_dir: Path of the local directory to upload.
        :param cloud_dir: Cloud destination directory. If not provided, upload to the root.
        :return: None
        """
        pass
    @abstractmethod
    def download_files(self, cloud_path: Union[str, Path], local_path: Optional[Union[str, Path]] = None):
        """
        Implement logic to download files
        :param cloud_path:
        :param local_path:
        :return:
        """
        pass