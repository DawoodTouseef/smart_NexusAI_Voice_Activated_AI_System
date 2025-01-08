from Crewai import Agent

from tools.calculator import CalculatorTool
from tools.sec import SEC10KTool, SEC10QTool

from crewai_tools import WebsiteSearchTool, ScrapeWebsiteTool


class StockAnalysisCrew:
    def __init__(self,llm):
        super().__init__()
        self.llm=llm
    def financial_agent(self) -> Agent:
        return Agent(
            role="The Best Financial Analyst",
            goal="Impress all customers with your financial data and market trends analysis",
            backstory="The most seasoned financial analyst with lots of expertise in stock market analysis and investment strategies that is working for a super important customer.",
            llm=self.llm,
            tools=[
                ScrapeWebsiteTool(),
                WebsiteSearchTool(
                    config=dict(
                        embedding_model=dict(
                            provider="google",
                            config=dict(
                                model="models/text-embedding-004",
                                task_type="retrieval_document",
                            )
                        )
                    )
                ),
                CalculatorTool(),
                SEC10QTool("AMZN",
                           config=dict(
                               embedding_model=dict(
                                   provider="google",
                                   config=dict(
                                       model="models/text-embedding-004",
                                       task_type="retrieval_document",
                                   )
                               )
                           )
                           ),

                SEC10KTool("AMZN",
                           config=dict(
                               embedding_model=dict(
                                   provider="google",
                                   config=dict(
                                       model="models/text-embedding-004",
                                       task_type="retrieval_document",
                                   )
                               )
                           )
                           ),

            ]
        )

    def research_analyst_agent(self) -> Agent:
        return Agent(
            role=" Staff Research Analyst",
            goal="Being the best at gathering, interpreting data and amazing your customer with it",
            backstory="Known as the BEST research analyst, you're skilled in sifting through news, company announcements,and market sentiments. Now you're working on a super important customer.",
            llm=self.llm,
            tools=[
                ScrapeWebsiteTool(),
                # WebsiteSearchTool(),
                SEC10QTool("AMZN",
                           config=dict(
                               embedding_model=dict(
                                   provider="google",
                                   config=dict(
                                       model="models/text-embedding-004",
                                       task_type="retrieval_document",
                                   )
                               )
                           )
                           ),
                SEC10KTool("AMZN",
                           config=dict(
                               embedding_model=dict(
                                   provider="google",
                                   config=dict(
                                       model="models/text-embedding-004",
                                       task_type="retrieval_document",
                                   )
                               )
                           )
                           ),
            ]
        )

    def financial_analyst_agent(self) -> Agent:
        return Agent(
            role="The Best Financial Analyst",
            goal="Impress all customers with your financial data and market trends analysis",
            backstory="The most seasoned financial analyst with lots of expertise in stock market analysis and investment strategies that is working for a super important customer.",
            llm=self.llm,
            tools=[
                ScrapeWebsiteTool(),
                WebsiteSearchTool(config=dict(
                        embedding_model=dict(
                            provider="google",
                            config=dict(
                                model="models/text-embedding-004",
                                task_type="retrieval_document",
                            )
                        )
                    )),
                CalculatorTool(),
                SEC10QTool(config=dict(
                        embedding_model=dict(
                            provider="google",
                            config=dict(
                                model="models/text-embedding-004",
                                task_type="retrieval_document",
                            )
                        )
                    )),
                SEC10KTool(config=dict(
                        embedding_model=dict(
                            provider="google",
                            config=dict(
                                model="models/text-embedding-004",
                                task_type="retrieval_document",
                            )
                        )
                    )),
            ]
        )

    def investment_advisor_agent(self) -> Agent:
        return Agent(
            role="Private Investment Advisor",
            goal="Impress your customers with full analyses over stocks and complete investment recommendations",
            backstory="You're the most experienced investment advisor and you combine various analytical insights to formulate strategic investment advice. You are now working for a super important customer you need to impress.",
            llm=self.llm,
            tools=[
                ScrapeWebsiteTool(),
                WebsiteSearchTool(config=dict(
                        embedding_model=dict(
                            provider="google",
                            config=dict(
                                model="models/text-embedding-004",
                                task_type="retrieval_document",
                            )
                        )
                    )),
                CalculatorTool(config=dict(
                        embedding_model=dict(
                            provider="google",
                            config=dict(
                                model="models/text-embedding-004",
                                task_type="retrieval_document",
                            )
                        )
                    )),
            ]
        )
