from Crewai import Task

from textwrap import dedent



class StockAnalysisCrew:

    def financial_analysis(self,agent) -> Task:
        return Task(
            agent=agent,
            description=dedent("""
            Conduct a thorough analysis of {company_stock}'s stock financial health and market performance. This includes examining key financial metrics such as
    P/E ratio, EPS growth, revenue trends, and debt-to-equity ratio. Also, analyze the stock's performance in comparison 
    to its industry peers and overall market trends.
            """),
            expected_output="""
            The final report must expand on the summary provided but now 
    including a clear assessment of the stock's financial standing, its strengths and weaknesses, 
    and how it fares against its competitors in the current market scenario.
    Make sure to use the most recent data possible.
            """
        )

    def research(self,agent) -> Task:
        return Task(
            agent=agent,
            description=dedent("""
            Collect and summarize recent news articles, press
    releases, and market analyses related to the {company_stock} stock and its industry.
    Pay special attention to any significant events, market sentiments, and analysts' opinions. 
    Also include upcoming events like earnings and others.
            """),
            expected_output="""
            Collect and summarize recent news articles, press
    releases, and market analyses related to the {company_stock} stock and its industry.
    Pay special attention to any significant events, market sentiments, and analysts' opinions. 
    Also include upcoming events like earnings and others. 
            """
        )

    def filings_analysis(self,agent) -> Task:
        return Task(
            agent=agent,
            description=dedent("""
            Analyze the latest 10-Q and 10-K filings from EDGAR for the stock {company_stock} in question. 
    Focus on key sections like Management's Discussion and analysis, financial statements, insider trading activity, 
    and any disclosed risks. Extract relevant data and insights that could influence
    the stock's future performance.
            """),
            expected_output="""
            Final answer must be an expanded report that now also highlights significant findings
    from these filings including any red flags or positive indicators for your customer.
            """
        )

    def recommend(self,agent) -> Task:
        return Task(
            agent=agent,
            description=dedent("""
            Review and synthesize the analyses provided by the
    Financial Analyst and the Research Analyst.
    Combine these insights to form a comprehensive
    investment recommendation. You MUST Consider all aspects, including financial
    health, market sentiment, and qualitative data from
    EDGAR filings. 
    
    Make sure to include a section that shows insider 
    trading activity, and upcoming events like earnings.
            """),
            expected_output="""
            Your final answer MUST be a recommendation for your customer. It should be a full super detailed report, providing a 
    clear investment stance and strategy with supporting evidence.
    Make it pretty and well formatted for your customer.
            """
        )


