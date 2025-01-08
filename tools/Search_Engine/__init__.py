import json
import os

import requests

from Crewai import Agent, Task
from Crewai.tools.base_tool import BaseTool
from unstructured.partition.html import partition_html


class BrowserTools(BaseTool):
    def __init__(self):
        super().__init__(
            name="Scrape website content",
            description="Useful to scrape and summarize a website content"
        )
    def _run(self,website):
        """Useful to scrape and summarize a website content"""
        #url = f"https://chrome.browserless.io/content?token={os.environ['BROWSERLESS_API_KEY']}"
        url=f"https://search.brave.com/search?q={website}"
        headers = {'cache-control': 'no-cache',
                   'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36'
                   }
        response = requests.request("POST", url, headers=headers)
        elements = partition_html(text=response.text)
        content = "\n\n".join([str(el) for el in elements])
        content = [content[i:i + 8000] for i in range(0, len(content), 8000)]
        summaries = []
        for chunk in content:
          agent = Agent(
              role='Principal Researcher',
              goal=
              'Do amazing researches and summaries based on the content you are working with',
              backstory=
              "You're a Principal Researcher at a big company and you need to do a research about a given topic.",
              allow_delegation=False)
          task = Task(
              agent=agent,
              description=
              f'Analyze and summarize the content bellow, make sure to include the most relevant information in the summary, return only the summary nothing else.\n\nCONTENT\n----------\n{chunk}'
          )
          summary = task.execute()
          summaries.append(summary)
        return "\n\n".join(summaries)