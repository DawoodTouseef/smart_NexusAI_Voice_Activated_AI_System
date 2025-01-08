from typing import Optional,Union,Dict,Any,List
import requests


class LLM:
    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        api_key: Optional[str] = None,
        format: Optional[str] = None,
        options: Optional[dict] = None,
        template: Optional[str] = None,
        stream: Optional[bool] = None,
        keep_alive: Optional[Union[int, str]] = None,
        callbacks: List[Any] = [],
        **kwargs,
    ):
        self.model = model
        self.base_url = base_url
        self.api_version = api_version
        self.api_key = api_key
        self.callbacks = callbacks
        self.kwargs = kwargs
        self.format=format
        self.options=options
        self.template=template
        self.stream=stream
        self.keep_alive=keep_alive

    def call(self, messages: List[Dict[str, str]], callbacks: List[Any] = [],files=None) -> str:
        """

        :param messages:
        :return:
        """
        try:
            params = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "metadata":
                    {
                        "chat_id": None,
                        "message_id": None,
                        "session_id": None,
                        "tool_ids":None,
                        "files": open(files,"wb") if files else None
                    }
                ,
            }

            # Remove None values to avoid passing unnecessary parameters
            params = {k: v for k, v in params.items() if v is not None}
            # Set the headers
            headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {self.api_key}',
            }
            # Build the URL
            url = f"{str(self.base_url).rstrip('/')}/api/chat/completions"
            response = requests.post(url, json=params, headers=headers)
            if response.status_code==200:
                return response.json()['choices'][0]['message']['content']
            else:
                pass
        except Exception as e:

            raise  # Re-raise the exception after logging

