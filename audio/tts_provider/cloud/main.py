from abc import abstractmethod


class Cloud:
    def __init__(self):
        super().__init__()

    @abstractmethod
    def check_files(self):
        pass
    @abstractmethod
    def upload(self,filename,remote_name):
        pass