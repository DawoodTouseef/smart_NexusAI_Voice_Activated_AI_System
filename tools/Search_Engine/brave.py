from langchain_community.utilities import BraveSearchWrapper
from Crewai.tools.base_tool import BaseTool
from langchain_community.tools import BraveSearch
from env import (
    BRAVE_SEARCH_MAX_RESULTS,
    BRAVE_SEARCH_URL,
    BRAVE_API_KEY,
    BRAVE_SAFE_SEARCH_RESULTS,
    BRAVE_COUNTRY,
    BRAVE_SEARCH_LANG,
    BRAVE_UI_LANG,
    BRAVE_SPELLCHECK,
    BRAVE_FRESHNESS,
    BRAVE_OFFSET,
)


class BraveSearchTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="Brave Search",
            description="A wrapper around Brave Search. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query or query and the total number of results to return."
        )
    def _run(self,data,k):
        """
        "A wrapper around Brave Search. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query or query and the total number of results to return."
        :param data:
        :return:
        """
        if k:
            max_result = k
        else:
            max_result = BRAVE_SEARCH_MAX_RESULTS
        parameters={
            "country":BRAVE_COUNTRY,
            "search_lang":BRAVE_SEARCH_LANG,
            "ui_lang":BRAVE_UI_LANG,
            "count":max_result,
            "offset":BRAVE_OFFSET,
            "safesearch":BRAVE_SAFE_SEARCH_RESULTS,
            "freshness":BRAVE_FRESHNESS,
            "spellcheck":BRAVE_SPELLCHECK,
            "result_filter":"""
                discussions,
                faq,
                infobox,
                news,
                query,
                summarizer,
                videos,
                web,
                locations,
            """
        }
        api_wrapper = BraveSearchWrapper(api_key=BRAVE_API_KEY,base_url=BRAVE_SEARCH_URL,
                                         search_kwargs=parameters)
        bing_search = BraveSearch(search_wrapper=api_wrapper)

        return bing_search.run(data)


