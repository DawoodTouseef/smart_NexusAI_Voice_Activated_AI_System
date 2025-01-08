from langchain.tools import tool
from numba.scripts.generate_lower_listing import description

from Crewai.tools .base_tool import BaseTool
from langchain_community.tools import BingSearchRun
from env import BING_SEARCH_URL,BING_SUBSCRIPTION_KEY,BING_SEARCH_MAX_RESULTS
from langchain_community.utilities import BingSearchAPIWrapper

class Bing(BaseTool):
    def __init__(self):
        super().__init__(
            name="Bing Search",
            description="A wrapper around Bing Search. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query or query and the total number of results to return."
        )
    def _run(self,data,k):
        """
        "A wrapper around Bing Search. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query or query and the total number of results to return."
        :param data:
        :return:
        """
        if k:
            max_result = k
        else:
            max_result = BING_SEARCH_MAX_RESULTS
        api_wrapper = BingSearchAPIWrapper(bing_subscription_key=BING_SUBSCRIPTION_KEY,bing_search_url=BING_SEARCH_URL,
                                           k=max_result)
        bing_search = BingSearchRun(api_wrapper=api_wrapper)

        return bing_search.run(data)


