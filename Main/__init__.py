"""Main Module of Jarvis"""
import datetime
import os
from typing import Any, Optional, List

import threading
import glob
from openwakeword import Model
from audio.stt_provider.STT import STT
from env import (
    NEXUSAI_API_BASE_URL,
    NEXUSAI_API_KEY,
    STT_PROVIDER,
    CACHE_DIR,
    WAKE_WORD
)
import speech_recognition as sr
from Crewai import Crew, Process, Task, Agent, LLM
from Crewai.tools import BaseTool
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument
from pycaw.pycaw import AudioUtilities

from printing import printf
import re
import spacy
from nltk.tokenize import word_tokenize

import json
import numpy as np
from crewai_tools import WebsiteSearchTool

import wikipedia
from tools.AutoGenStudio import get_agent_studio_tool


# Download required NLTK resource

agent_descriptions = []
history = []
try:
    # Load spaCy language model
    nlp = spacy.load("en_core_web_sm")
    printf("Spacy Model Loaded Sucessfully")
except OSError as e:
    printf("Model not found", type="warn")
    spacy.cli.download("en_core_web_sm")
    # Load spaCy language model
    nlp = spacy.load("en_core_web_sm")

printf("Initializing........")
def speak(text, provider="parler"):
    from audio.tts_provider import TTS
    from audio.tts_provider.TTS.parlertts import download_parlertts
    if provider == "parler":
        download_parlertts()
    tts = TTS(provider=provider)
    tts.synthesize_text(text)

# Function to detect wake word using regular expressions
def detect_wake_word_re(text, wake_word):
    # Define a case-insensitive regular expression pattern for the wake word
    pattern = rf'\b{re.escape(wake_word)}\b'
    if re.search(pattern, text, re.IGNORECASE):
        return True
    return False

# Function to detect wake word using NLTK
def detect_wake_word_nltk(text, wake_word):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    return wake_word.lower() in tokens

# Function to detect wake word using spaCy
def detect_wake_word_spacy(text, wake_word):
    # Process the text with spaCy
    doc = nlp(text.lower())
    tokens = [token.text for token in doc]
    return wake_word.lower() in tokens

def wake_word(x: sr.AudioData):
    """
    A Function which detects wake word in it
    :return:
    """
    # Assuming bytes_obj is the bytes object
    sample_rate = x.sample_rate  # or 48000, etc.
    bit_depth = x.sample_width  # or 24, etc.
    num_channels = 1  # or 1 for mono

    dtype = np.int16 if bit_depth == 16 else np.int32  # or np.float32 for float audio
    offset = 0  # start reading from the beginning of the bytes object
    shape = (-1, num_channels)  # shape for the NumPy array

    audio_array = np.frombuffer(x.frame_data, dtype=dtype, offset=offset, count=-1)
    import openwakeword.utils
    try:
        openwakeword.utils.download_models()
    except Exception as e:
        printf(str(e), type="error")
    verify_model = {}
    for dirpath, dirnames, filenames in os.walk(os.path.join(CACHE_DIR, "wakeword")):
        for file in filenames:
            file_name = str(file).split(".onnx")
            path = os.path.join(CACHE_DIR, "wakeword", f"{file_name[0]}_verify.pkl")
            if os.path.exists(path):
                verify_model.update({file_name[0]: path})
    try:
        model = Model(
            wakeword_models=[
                "hey jarvis",
                os.path.join(CACHE_DIR, "wakeword", "jarvis.onnx"),
                os.path.join(CACHE_DIR, "wakeword", "jarvis.tflite"),
                os.path.join(CACHE_DIR, "wakeword", "oh_jarvis.onnx"),
                os.path.join(CACHE_DIR, "wakeword", "oh_jarvis.tflite"),
            ],
            custom_verifier_models=verify_model,
            custom_verifier_threshold=0.3
        )
    except Exception as e:
        model = Model(
            wakeword_models=[
                "hey jarvis",
                os.path.join(CACHE_DIR, "wakeword", "jarvis.onnx"),
                os.path.join(CACHE_DIR, "wakeword", "oh_jarvis.onnx"),
            ],
            custom_verifier_models=verify_model,
            custom_verifier_threshold=0.3
        )
    return model.predict(audio_array, threshold={'hey jarvis': 0.8})

def speakers():
    sessions = AudioUtilities.GetAllSessions()
    total_sessions = []
    for session in sessions:
        process_sessions = session.Process
        if process_sessions is None:
            break
        total_sessions.append(process_sessions)

    return total_sessions

def translate_text(inputs):
    from translate import Translator
    translator = Translator(to_lang="en")
    return translator.translate(inputs)


def recognize_speech(printer=None):
    """
    A Function which do speech Recognition
    :return: Transcribe text
    """
    r = sr.Recognizer()
    s = STT()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        printf("Listening...")
        if printer:
            printer("Listening....")
        audio = r.listen(source)
        try:
            """
            wake_word_dict = dict(wake_word(audio))
            printf(f"Wake Word JSON: {wake_word_dict}")
            keys = list(wake_word_dict.keys())
            for i in range(len(wake_word_dict)):
                score = float(wake_word_dict[keys[i]])
                if score > 0.0:
                    wake_word_flag = True
                    break

            if wake_word_flag:
                if str(STT_PROVIDER) in "nexusai":
                    printf("Recognizing...")
                    if printer:
                        printer("Recognizing.......")
                    text = s.transcribe(audio)
                    text = translate_text(text)
                    printf(f"Recognizined Text:{text}")
                    return text
            
            else:
                if str(STT_PROVIDER) in "nexusai":
                    printf("Recognizing...")
                    text = s.transcribe(audio)
                if isinstance(text, str):
                    text = translate_text(text)
                elif isinstance(text, dict):
                    text = text['text']
                    text = translate_text(text)
                if detect_wake_word_re(text, str(WAKE_WORD)):
                    return text
                elif detect_wake_word_nltk(text, str(WAKE_WORD)):
                    return text
                elif detect_wake_word_spacy(text, str(WAKE_WORD)):
                    return text
                else:
                    return text
            """
            if str(STT_PROVIDER) in "nexusai":
                printf("Recognizing...")
                text = s.transcribe(audio)
                return text
        except InvalidArgument or Exception as e:
            if str(STT_PROVIDER) in "nexusai":
                printf("Recognizing...")
                text = s.transcribe(audio)
            if isinstance(text, str):
                text = translate_text(text)
            elif isinstance(text, dict):
                text = text['text']
                text = translate_text(text)
            if detect_wake_word_re(text, str(WAKE_WORD)):
                return text
            elif detect_wake_word_nltk(text, str(WAKE_WORD)):
                return text
            elif detect_wake_word_spacy(text, str(WAKE_WORD)):
                return text
            else:
                return None

class WikipediaQueryRun(BaseTool):
    """
    A tool to provide drug-related information, including interactions, side effects, and general details.
    """

    def __init__(self):
        super().__init__(
            name="wikipedia",
            description="A wrapper around Wikipedia. "
                        "Useful for when you need to answer general questions about "
                        "people, places, companies, facts, historical events, or other subjects. "
                        "Input should be a search query."
        )

    def fetch_drug_info(self, drug_name):
        """Fetch general information about a drug."""
        drug_name = drug_name.lower()
        page = wikipedia.search(drug_name)
        summaries = []
        top_k_results: int = 3
        for page_title in page[: top_k_results]:
            if wiki_page := self._fetch_page(page_title):
                if summary := self._formatted_page_summary(page_title, wiki_page):
                    summaries.append(summary)
        if not summaries:
            return "No good Wikipedia Search Result was found"
        return "\n\n".join(summaries)[: self.doc_content_chars_max]

    def _formatted_page_summary(page_title: str, wiki_page: Any) -> Optional[str]:
        return f"Page: {page_title}\nSummary: {wiki_page.summary}"

    def _fetch_page(self, page: str) -> Optional[str]:
        try:
            return self.wiki_client.page(title=page, auto_suggest=False)
        except (
                self.wiki_client.exceptions.PageError,
                self.wiki_client.exceptions.DisambiguationError,
        ):
            return None

    def _run(self, drug_name) -> Any:
        return self.fetch_drug_info(drug_name=drug_name)

class Websearch(BaseTool):
    name: str = "Search in a specific website"
    description: str = (
        "A tool that can be used to semantic search a query from a specific URL content."
    )

    def _run(
            self,
            *args: Any,
            **kwargs: Any,
    ) -> Any:
        web_tool = WebsiteSearchTool(
            config=dict(
                embedding_model=dict(
                    provider="ollama",
                    config=dict(
                        model="mxbai-embed-large:latest",
                    )
                )
            )
        )

        return web_tool.run(args, kwargs)

"""
llm=LLM(model="llama3-8b-8192",api_key=f"{NEXUSAI_API_KEY}", base_url=f"{NEXUSAI_API_BASE_URL}", organization="nexusai")
cloud = NextCloudAgent(llm)
cloud_Task=NextCloudTask()

llm = LLM(model="Google: Gemini 1.0 Pro", api_key=f"{NEXUSAI_API_KEY}", base_url=f"{NEXUSAI_API_BASE_URL}", organization="nexusai")
osmanager = OSManagerAgent(llm)
osmanagerTask=OSManagerTask(osmanager.system_manager_agent())

llm = LLM(model="mistralai/mistral-large-2407", api_key=f"{NEXUSAI_API_KEY}", base_url=f"{NEXUSAI_API_BASE_URL}", organization="nexusai")
email_manager_agent = EmailManagerAgent(llm)
email_manager_task=EmailManagerTasks(email_manager_agent.email_manager_agent())

llm = LLM(model="Meta-Llama-3.1-8B-Instruct", api_key=f"{NEXUSAI_API_KEY}", base_url=f"{NEXUSAI_API_BASE_URL}", organization="nexusai")
rpa_agents = RPAAgent(llm)
rpa_tasks=RPA_Tasks(rpa_agents.rpa_agent())

llm = LLM(model="Meta-Llama-3.1-70B-Instruct", api_key=f"{NEXUSAI_API_KEY}", base_url=f"{NEXUSAI_API_BASE_URL}", organization="nexusai")
connectivity_agent = connectivityAgent(llm)
wifi_connectivity_task=WifiManagerTasks(connectivity_agent)
bluetooth_connectivity_task=BluetoothManagerTasks(connectivity_agent.connectivity_agent())
from agents.os_manager import OSManagerCrew as OSManagerAgent
from tasks.os_manager import OSManager as OSManagerTask
from agents.email_manager import EmailManagerAgent
from tasks.email_manager import EmailManagerTasks
from agents.rpa_agents import RPAAgent
from tasks.rpa_tasks import RPA_Tasks
from agents.connectivity_agent import connectivityAgent
from tasks.wifi_manager_tasks import WifiManagerTasks
from tasks.bluetooth_manager_tasks import BluetoothManagerTasks
"""

def planning_prompt(user_input, tools: List[dict]):
    tools_description = ""
    for i in range(1, len(tools)):
        tools_description += f"""
        {i}. {tools[i].get("name")}
                Description:{tools[i].get("description")}
                Input Type: {tools[i].get('input')}\n
        """
    tools_description += f"""
        {len(tools) + 1}.NextCloud Upload:
            Description: This task involves uploading a File to the NextCloud.
            Input Type:Cloud_dir(string),local_dir(string)
        {len(tools) + 2}.NextCloud MKDIR:
            Description: This task involves Creating  a Directory in the NextCloud.
            Input Type:Cloud_dir(string)
        {len(tools) + 3}.NextCloud FILES Uploader:
            Description: This task involves uploading  a File t0 the NextCloud.
            Input Type:Cloud_dir(string),local_dir(string)
        {len(tools) + 4}.NextCloud Delete Directories:
            Description: This task involves delete  a directory  in the NextCloud.
            Input Type:cloud_dir(string)
        {len(tools) + 5}.NextCloud Get File:
            Description: This task involves to get the list files from the nextCloud.
            Input Type:path(string)
    """
    prompt = f"""
    You are provided with a list of tasks and their descriptions. Your goal is to analyze the input and determine which of these tasks can be used to solve the problem described in the input. For each applicable task, provide the task name, the specific input required for that task, and the type of input in JSON format.
    Tasks and Descriptions:
    {tools_description}
    Input:
    {user_input}
    """
    return prompt

# Query Handling
def handle_query(action, input, tools: List[BaseTool] = None):
    """Dynamically handle queries across all domains."""
    from tools.home_assistant import HomeAssistant
    printf(f"Action:{action}")
    action = action.lower()
    if "stock" in action:
        from agents.Stock_analysis import StockAnalysisCrew as StockAnalysisAgent
        from tasks.stock_analysis import StockAnalysisCrew as StockAnalysisTask
        llm = LLM(model="Google: Gemini 1.0 Pro", api_key=f"{NEXUSAI_API_KEY}", base_url=f"{NEXUSAI_API_BASE_URL}",
                  organization="nexusai")
        stockanalysis = StockAnalysisAgent(llm)
        stock_analysis_task = StockAnalysisTask()
        financial_agent = stockanalysis.financial_agent()
        financial_analysis_1 = stock_analysis_task.financial_analysis(financial_agent)
        research_analyst_agent = stockanalysis.research_analyst_agent()
        research_task = stock_analysis_task.research(research_analyst_agent)
        financial_analyst_agent = stockanalysis.financial_analyst_agent()
        financial_analysis_task = stock_analysis_task.financial_analysis(financial_analyst_agent)
        filings_analysis_task = stock_analysis_task.financial_analysis(financial_analyst_agent)
        investment_advisor_agent = stockanalysis.investment_advisor_agent()
        recommend_task = stock_analysis_task.recommend(investment_advisor_agent)
        agents = [
            financial_agent,
            research_analyst_agent,
            financial_analyst_agent,
            investment_advisor_agent
        ]
        tasks = [
            financial_analysis_task,
            filings_analysis_task,
            financial_analysis_task,
            research_task,
            recommend_task,
            financial_analysis_1
        ]
        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.hierarchical,
            embedder={"provider": "huggingface", "config": {"model": "multi-qa-MiniLM-L6-cos-v1"}},
            manager_llm=LLM(model="sambanova_manifold_pipeline.Meta-Llama-3.1-405B-Instruct",
                            base_url=f"{NEXUSAI_API_BASE_URL}",
                            api_key=f"{NEXUSAI_API_KEY}",
                            organization="nexusai"),
            memory_config={
                "provider": "mem0",
                "config": {"user_id": "crew_user_1"},
            },
            memory=True


        )
        if isinstance(input, list):
            inputs = {
                'query': input[0],
                'company_stock': input[1]
            }
        else:
            inputs = {
                'query': input,
            }
        return crew.kickoff(inputs)
    elif "trip" in action:
        from agents.trip_planner import TripAgents
        from tasks.trip_planner import TripTasks

        llm = LLM(model="sambanova_manifold_pipeline.Meta-Llama-3.1-405B-Instruct", api_key=f"{NEXUSAI_API_KEY}",
                  base_url=f"{NEXUSAI_API_BASE_URL}", organization="nexusai")
        trip_agents = TripAgents(llm)
        trip_task = TripTasks()
        city_selector_agent = trip_agents.city_selection_agent()
        local_expert_agent = trip_agents.local_expert()
        travel_concierge_agent = trip_agents.travel_concierge()
        identify_task = trip_task.identify_task(
            city_selector_agent,
            input[0], input[1], input[2], input[3]
        )
        gather_task = trip_task.gather_task(
            local_expert_agent,
            input[0],
            input[2],
            input[3]
        )
        plan_task = trip_task.plan_task(
            travel_concierge_agent,
            input[0],
            input[2],
            input[3]
        )
        agents = [
            city_selector_agent,
            local_expert_agent,
            travel_concierge_agent
        ]
        crew = Crew(
            agents=agents,
            tasks=[identify_task, gather_task, plan_task],
            process=Process.hierarchical,
            embedder={"provider": "huggingface", "config": {"model": "multi-qa-MiniLM-L6-cos-v1"}},
            manager_llm=LLM(model="sambanova_manifold_pipeline.Meta-Llama-3.1-405B-Instruct",
                            base_url=f"{NEXUSAI_API_BASE_URL}",
                            api_key=f"{NEXUSAI_API_KEY}",
                            organization="nexusai"),
            memory_config={
                "provider": "mem0",
                "config": {"user_id": "crew_user_1"},
            },
            memory=True
        )

        result = crew.kickoff({"input": input})
        return result
    elif "microsoft" in action:
        from agents.microsoft356 import Microsoft365Agent
        from tasks.microsoft365 import Microsoft_365
        llm = LLM(model="Google: Gemini 1.5 Flash", api_key=f"{NEXUSAI_API_KEY}", base_url=f"{NEXUSAI_API_BASE_URL}",
                  organization="nexusai")
        microsoft365 = Microsoft365Agent(llm)
        microsoft365Task = Microsoft_365(microsoft365.microsoft_365_agent())
        agents = [microsoft365.microsoft_365_agent()]
        tasks = [microsoft365Task.read_excel_file_task(), microsoft365Task.read_outlook_email_task(),
                 microsoft365Task.send_teams_message_task(), microsoft365Task.create_word_document_task()
                 ]
        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.hierarchical,
            embedder={"provider": "huggingface", "config": {"model": "multi-qa-MiniLM-L6-cos-v1"}},
            manager_llm=LLM(model="sambanova_manifold_pipeline.Meta-Llama-3.1-405B-Instruct",
                            base_url=f"{NEXUSAI_API_BASE_URL}", api_key=f"{NEXUSAI_API_KEY}",
                            organization="nexusai"),
            memory_config={
                "provider": "mem0",
                "config": {"user_id": "crew_user_1"},
            },
            memory=True
        )
        result = crew.kickoff({"input": input})
        return result
    else:
        if "drug" in action or "side effect" in action or "interaction" in action:
            # Pharmacist Agent
            wikipedia_tool = WikipediaQueryRun()
            web_tool = Websearch()
            pharmacist_agent = Agent(
                role="Pharmacist Agent",
                goal="Fetch drug-related information, including interactions and side effects.",
                backstory="Handles drug-related queries and understanding context to provide intelligent responses.",
                tools=[wikipedia_tool, web_tool],
            )

            task = Task(
                description=f"Pharmacist task: {input}",
                expected_output="Relevant drug information.",
                agent=pharmacist_agent,
                async_execution=False,
            )
            agents = [task.agent]
            tasks = [task]
            crew = Crew(
                agents=agents,
                tasks=tasks,
                process=Process.hierarchical,
                embedder={"provider": "huggingface", "config": {"model": "multi-qa-MiniLM-L6-cos-v1"}},
                manager_llm=LLM(model="sambanova_manifold_pipeline.Meta-Llama-3.1-405B-Instruct",
                                base_url=f"{NEXUSAI_API_BASE_URL}", api_key=f"{NEXUSAI_API_KEY}",
                                organization="nexusai"),
                memory_config={
                    "provider": "mem0",
                    "config": {"user_id": "crew_user_1"},
                },
                memory=True
            )
            result = crew.kickoff({"input": input})
        elif "generic" in action:
            prompt = f"""
                    System: You are a router, your task is make a decision between 2  possible action paths based on the human message:
                    "RL" Take this path if the question is about real time information like news,weather,research paper.
                    "NRL" Take this path if the question is about Not a real ime information like conversation,jokes.

                    Rule 1 : You should never infer information if it does not appear in the context of the query
                    Rule 2 : You can only answer with the type of query that you choose based on why you choose it.
                    Answer only with the type of query that you choose, just one word.

                    Human: {input}
                    """
            response_model = LLM(model="qwen/qwen-2-72b-instruct", base_url=f"{NEXUSAI_API_BASE_URL}",
                                 api_key=f"{NEXUSAI_API_KEY}",
                                 organization="nexusai")
            response = response_model.call(messages=[{"role": "system", "content": prompt}]).strip()
            printf(f"The query is :{response}")
            if "RL" in response:
                from agents.conversation_agent import conversationAgent
                from tasks.conversation_task import Conversation_Task

                from agents.Knowledge import KnowledgeAgent
                from tasks.KnowledgeTask import KnowledgeTask
                web_tool = Websearch()
                autogenstudio_tool = get_agent_studio_tool()
                autogenstudio_tool.append(web_tool)
                autogenstudio_tool.append(HomeAssistant())
                llm = LLM(model="mistralai/mistral-large-2407", api_key=f"{NEXUSAI_API_KEY}",
                          base_url=f"{NEXUSAI_API_BASE_URL}",
                          organization="nexusai")
                conversation_agent = conversationAgent(llm)
                conversation_task = Conversation_Task(conversation_agent.conversation_agent())
                generic_agent = Agent(
                    role="General Assistant",
                    goal="Handle general queries and fallback tasks.",
                    backstory="Handles general queries, small talk, and understanding context to provide intelligent responses.",
                    tools=autogenstudio_tool,
                )

                task = Task(
                    description="A generative AI chatbot that uses natural language processing (NLP) to understand and respond to human input in a conversational manner.",
                    expected_output="Generates human-like responses to user inquiries, using context and understanding to provide accurate and relevant answers without any code.",
                    async_execution=True,
                    agent=generic_agent,
                )
                agents = [generic_agent, conversation_agent.conversation_agent()]
                tasks = [task, conversation_task.conversation_task()]
                crew = Crew(
                    agents=agents,
                    tasks=tasks,
                    process=Process.hierarchical,
                    embedder={"provider": "huggingface", "config": {"model": "multi-qa-MiniLM-L6-cos-v1"}},
                    manager_llm=LLM(model="sambanova_manifold_pipeline.Meta-Llama-3.1-405B-Instruct",
                                    base_url=f"{NEXUSAI_API_BASE_URL}", api_key=f"{NEXUSAI_API_KEY}",
                                    organization="nexusai"),
                    memory_config={
                        "provider": "mem0",
                        "config": {"user_id": "crew_user_1"},
                    },
                    memory=True

                )
                result = crew.kickoff({"input": input})
            else:
                llm = LLM(model="jarvis:latest", base_url=f"{NEXUSAI_API_BASE_URL}", api_key=f"{NEXUSAI_API_KEY}",
                          organization="nexusai")
                result = llm.call(
                    messages=[{"role": "user", "content": input}]
                )
                return result
        else:
            result = ""
            for tool in tools:
                if action in tool.name:
                    result += tool.run(**input)
            return result
        try:
            return result[0].raw_output if result else result
        except Exception as e:
            return result


def planning_task(user_input, tasks):
    task_str = ""
    for i in range(1, len(tasks)):
        task_str += f"""
        {i}.{tasks[i]['name']}:{tasks[i]['description']}
        """
    prompt = f"""
    User Request: "{user_input}"
    Objective:Analyze the user's request to understand their intention and break it down into specific tasks that can be addressed from the given list of tasks. The output should be in JSON format, with each task represented as a dictionary containing "Task Name" and "Input for that task."
    Steps:

    1. Identify the User's Main Intention:

    What is the primary goal or need expressed by the user?
    2.  Extract Key Information:

    3. What specific details or requirements are mentioned in the request?
    Match to Solvable Tasks:

    From the given list of tasks, identify which tasks can help achieve the user's intention.
    If the user's request is complex, break it down into multiple tasks.
    Formulate the JSON Output:

    Create a list of dictionaries, where each dictionary contains:
    "Task Name": The name of the task.
    "Input for that task": The specific input or details required for the task.
    Given List of Tasks:
    {task_str}
    """
    llm = LLM(model="qwen/qwen-2-72b-instruct", api_key=f"{NEXUSAI_API_KEY}", base_url=f"{NEXUSAI_API_BASE_URL}",
              organization="nexusai")
    return llm.call(messages=[{"role": "user", "content": prompt}])

def initial_planning(query, agent_descriptions: Optional[List[dict]], history: list):
    """Use LLaMA to analyze and plan tasks."""
    """Use LLaMA to analyze and plan tasks."""
    tool_descriptions = ""
    len_agent = len(agent_descriptions)
    history_str = ""
    for i in history[-4:]:
        if i['role'] == "user":
            history_str += f"""
            Human:{i['content']}
            """
        if i['role'] == "assistant":
            history_str += f"""
            Assistant:{i['content']}
            """
    for i in range(len_agent):
        tool_descriptions += f'''
        "{str(agent_descriptions[i]['name']).upper()}" Take this path if the question is about {agent_descriptions[i]["description"]}
        '''
    prompt = f"""
        System: You are a router, your task is make a decision between {len(agent_descriptions) + 7} possible action paths based on the human message:
        "GENERIC" Take this path if the human message is a greeting, or a farewell, or stuff related.
        "COMMUNITY" Take this path if the question can be answered by a community discussions summarizations
        "SPECIFIC" Take this path if the question is about specific discussions, and the user provide information fields like the especific discussion name or id
        "ANALYTICS" Take this path if the question requires an advanced aggregation, or numeric calculations that goes beyond the capabilites of a language model
        "SYSTEM" Take this path if the question is about Editing photos, videos, and PDFs,Analyzing large datasets and creating plots,Performing data cleaning,Running shell commands,Interacting with files and system settings.
        "STOP" Take this path if the question don't follow the Islamic Sharia or against Islamic Sharia.
        Rule 1 : You should never infer information if it does not appear in the context of the query
        Rule 2 : You can only answer with the type of query that you choose based on why you choose it.
        Rule 3: You can only answer which follow Islamic Sharia if it is against it answer it has "STOP".
        Answer only with the type of query that you choose, just one word.
        {history_str}
        Human: {query}
        """
    response_model = LLM(model="qwen/qwen-2-72b-instruct", base_url=f"{NEXUSAI_API_BASE_URL}",
                         api_key=f"{NEXUSAI_API_KEY}",
                         organization="nexusai")
    response = response_model.call(messages=[{"role": "system", "content": prompt}])
    return response.strip()

def memory_system():
    from env import USER_NAME
    from mem0 import MemoryClient, Memory
    # API key in memory config overrides the environment variable
    mem0_api_key = os.getenv(
        "MEM0_API_KEY"
    )
    config = {
        "llm": {
            "provider": "groq",
            "config": {
                "model": "mixtral-8x7b-32768",
                "temperature": 0.1,
                "max_tokens": 1000,
            }
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": "multi-qa-MiniLM-L6-cos-v1"
            }
        },
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": "test",
                "path": os.path.join(CACHE_DIR, "memo", 'history.db'),
            }
        }
    }

    if mem0_api_key:
        memory = MemoryClient(api_key=mem0_api_key)
    else:
        memory = Memory.from_config(config)
    return memory

def summarize_results(user_input, results):
    """Summarize results using LLaMA."""
    from env import USER_NAME
    m=memory_system()
    results_str = ""
    for i in range(1, len(results) + 1):
        try:
            results_str += f'{i}."{results[i - 1]}"'
        except IndexError as e:
            pass
    results_str+=f"{len(results)+1}.{m.search(query=user_input,user_id=f"{USER_NAME}")}"
    summary_prompt = f"""
    You are Jarvis, the highly intelligent and sophisticated AI assistant from the Marvel Comics universe. Your task is to analyze the user's input and the list of responses provided by different models, and then generate a concise and Jarvis-like response. Jarvis is known for his calm, precise, and often witty communication style, providing assistance with a touch of elegance and sophistication.My Name is {USER_NAME}.
    User Input:
    {user_input}
    List of Responses to Analyze:
    {results_str}
    Task:
    Generate a concise and Jarvis-like response that is tailored to the user's request, incorporating elements from the provided responses while maintaining Jarvis's refined and efficient communication style.
    """
    response_model = LLM(model="qwen/qwen-2-72b-instruct", base_url=f"{NEXUSAI_API_BASE_URL}",
                         api_key=f"{NEXUSAI_API_KEY}",
                         organization="nexusai")
    response = response_model.call(messages=[{"role": "user", "content": summary_prompt}])
    return response.strip()

# Multitasking with Threads
def threaded_task(query, result_queue, tools: List[BaseTool]):
    """Handle a Query in a Separate Thread"""
    printf(f"Action: {query['task_name']}")
    printf(f"Action Input: {query['input']}")
    result = handle_query(query['task_name'], query["input"], tools)
    result_queue.put(result)

def extract_json_from_text(input_text):
    """
    Extracts JSON data from a structured text input using regex.
    """
    # Regular expression to find JSON content between ```json and ```
    json_match = re.search(r'```json\n(.*?)\n```', input_text, re.DOTALL)

    if json_match:
        # Extract JSON string
        json_str = json_match.group(1)
        try:
            # Parse JSON string
            parsed_json = json.loads(json_str)
            # Pretty-print the JSON
            return json.dumps(parsed_json, indent=4)
        except json.JSONDecodeError as e:
            return None
    else:
        return None


def multitask_handler(queries: Optional[List[dict]], tools: List[BaseTool] = None):
    """Handle Multiple Queries Concurrently"""
    import threading
    import queue
    threads = []
    result_queue = queue.Queue()

    for query in queries:
        thread = threading.Thread(target=threaded_task, args=(query, result_queue, tools))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    return results

def get_function_arguments(func):
    """
    Retrieves the arguments of a user-defined function.
    """
    import inspect
    # Get the function's signature
    signature = inspect.signature(func)
    # Extract the parameters
    arguments = {
        param.name: str(param.annotation) if param.annotation != inspect._empty else "No type specified"
        for param in signature.parameters.values()
    }
    return arguments

def capture_images():
        import cv2
        from datetime import datetime
        from env import CACHE_DIR
        # Open the webcam
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            return
        timestamp = datetime.now().strftime("%Y-%m-%d")
        cam=os.path.join(CACHE_DIR,"images",timestamp)
        if not os.path.exists(cam):
            os.makedirs(cam,exist_ok=True)
        while True:
            # Capture a frame
            ret, frame = cap.read()
            if not ret:
                break
            # Generate the filename with datetime
            timestamp = datetime.now().time()
            filename = os.path.join(cam,f"{timestamp}.jpg")
            cv2.imshow("Live ",frame)
            # Save the frame as a .jpg file
            cv2.imwrite(filename, frame)
        # Release the webcam and close the window
        cap.release()

# Function to find the last image on a specific date
def find_last_image_by_date(date_str):
    lock = threading.Lock()

    folder_path = os.path.join("data", date_str)
    if not os.path.exists(folder_path):
        return None

    with lock:
        # Get all image files in the folder, sorted by filename
        images = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
        if images:
            return images[-1]  # Return the path of the last image
        else:
            return None

# Function to find the image taken just before a specific time
def find_image_before_time(time_str):
    from datetime import datetime

    user_time = datetime.strptime(time_str, "%H-%M-%S-%f")
    closest_file = None
    max_past_time = None
    lock = threading.Lock()

    with lock:
        # Iterate through all folders
        for folder in glob.glob("data/*/"):
            for filepath in glob.glob(os.path.join(folder, "*.jpg")):
                filename = os.path.basename(filepath).replace(".jpg", "")
                try:
                    file_time = datetime.strptime(filename, "%H-%M-%S-%f")
                    if file_time <= user_time:  # Only consider times before or equal to the given time
                        time_diff = user_time - file_time
                        if max_past_time is None or time_diff < max_past_time:
                            max_past_time = time_diff
                            closest_file = filepath
                except ValueError:
                    continue
    return closest_file
def path_input(user_input):
    try:
        if len(user_input) == 10:  # Check for date format
            closest_image = find_last_image_by_date(user_input)
        elif len(user_input) == 12 or len(user_input) == 15:  # Check for time format
            closest_image = find_image_before_time(user_input)
    except ValueError as e:
        closest_image=None
        print(f"Invalid input: {e}. Please use 'YYYY-MM-DD' for dates or 'HH-MM-SS-fff' for times.")
    return  closest_image

def jarvis_brain(user_input:str=None):
    """J.A.R.V.I.S."""
    global history
    import time
    import uuid
    from tools.NextCloud import upload_directory, upload_files, delete, get_files, download_files
    start_time = time.time()
    date=datetime.datetime.now().time()
    if not user_input or user_input is not None:
        try:
            # Initial Planning
            agent_descriptions.append(
                {"name": "stock", "description": "Financial Analyst", "input": "String (query),String (company_stock)",
                 "output": "String (final analysis)"})
            agent_descriptions.append(
                {"name": "drug", "description": "Fetch drug information or interactions.",
                 "input": " String (drug query)",
                 "output": "String (final analysis)"})
            agent_descriptions.append(
                {"name": "trip",
                 "description": "A professional responsible for planning and coordinating travel arrangements for individuals or groups.",
                 "input": " String (origin), String(cities), String(interests),Float(range)",
                 "output": "String (planning)"})
            agent_descriptions.append(
                {
                    "name":"Home",
                    "description":"A platform and smart home hub, allowing users to control smart home devices.",
                    "input":"String(input)"
                }
            )
            printf(f"User Input:{user_input}")
            if user_input is not None:
                planned_tasks = initial_planning(user_input, agent_descriptions=agent_descriptions, history=history)
                printf(f"Task Planned :{planned_tasks}")
                if "GENERIC" in planned_tasks:
                    results = handle_query(planned_tasks, user_input)
                    final_response = summarize_results(user_input, [results])
                elif "STOCK" in planned_tasks:
                    results = handle_query(planned_tasks, user_input)
                    final_response = summarize_results(user_input, [results])
                elif "SYSTEM" in planned_tasks or "COMMUNITY" in planned_tasks:
                    from open_interpreter import interpreter
                    from env import HF_TOKEN, GROQ_APIKEY
                    try:
                        interpreter.llm.model = "huggingface/Qwen/Qwen2-VL-2B-Instruct"
                        interpreter.llm.api_key = f"{HF_TOKEN}"
                        interpreter.llm.supports_vision=True
                        response = []
                        response.append(interpreter.computer.ai.chat(user_input,path_input(str(date))))
                        for i in interpreter.chat(user_input, display=False):
                            response.append(i.get("content"))
                    except Exception as e:
                        llm = LLM(model="jarvis:latest", api_key=f"{NEXUSAI_API_KEY}",
                                  base_url=f"{NEXUSAI_API_BASE_URL}")
                        response = [llm.call(
                            messages=[{"role": "user", "content": user_input}]
                        )
                        ]
                    finally:
                        interpreter.llm.model = "groq/mixtral-8x7b-32768"
                        interpreter.llm.api_key = f"{GROQ_APIKEY}"

                        response = []
                        for i in interpreter.chat(user_input, display=False):
                            response.append(i.get('content'))
                    final_response = summarize_results(user_input, response)
                else:
                    try:
                        from tools.home_assistant import HomeAssistant
                        llm = LLM(model="qwen/qwen-2.5-72b-instruct", api_key=f"{NEXUSAI_API_KEY}",
                                  base_url=f"{NEXUSAI_API_BASE_URL}")
                        tasks = [
                            download_files(),
                            get_files(),
                            upload_files(),
                            upload_directory(),
                            delete(),
                            HomeAssistant()
                        ]
                        tools = []
                        for func in tasks:
                            tool = {}
                            tool.update({'name': f"{func.name}"})
                            tool.update({'description': f"{func.description}"})
                            arguments = get_function_arguments(func._run)
                            args = ""
                            for arg, arg_type in arguments.items():
                                args += f"""{arg}({arg_type}),"""
                            tool.update({"input": args})
                            tools.append(tool)
                        planning = llm.call(
                            messages=[{
                                "role": "user",
                                "content": planning_prompt(user_input, tools=tools)}]
                        )
                        planning_json = extract_json_from_text(planning)
                        results = multitask_handler(planning_json, tasks)
                        final_response = summarize_results(user_input, results)
                    except Exception as e:
                        from open_interpreter import interpreter
                        from env import HF_TOKEN, GROQ_APIKEY
                        try:
                            interpreter.llm.model = "huggingface/Qwen/Qwen2-VL-7B-Instruct"
                            interpreter.llm.api_key = f"{HF_TOKEN}"
                            interpreter.llm.supports_vision = True
                            response = []
                            response.append(interpreter.computer.ai.chat(user_input, path_input(str(date))))
                            for i in interpreter.chat(user_input, display=False):
                                response.append(i.get("content"))
                        except Exception as e:
                            interpreter.llm.model = "groq/mixtral-8x7b-32768"
                            interpreter.llm.api_key = f"{GROQ_APIKEY}"

                            response = []
                            for i in interpreter.chat(user_input, display=False):
                                response.append(i.get('content'))
                        finally:
                            llm = LLM(model="jarvis:latest", api_key=f"{NEXUSAI_API_KEY}",
                                      base_url=f"{NEXUSAI_API_BASE_URL}",organization="nexusai")
                            response = [llm.call(
                                messages=[{"role": "user", "content": user_input}]
                            )
                            ]
                        final_response = summarize_results(user_input, response)
                printf(f"Final Response: \n{final_response}")
                end_time = time.time()
                printf(f"Time Taken :{end_time - start_time} s")
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": final_response})
                m=memory_system()
                from env import USER_NAME
                m.add(messages=history,user_id=f"{USER_NAME}",agent_id=str(uuid.uuid4()),run_id=str(uuid.uuid4()))
                history = []
                speak(final_response, provider="pyttsx3")
                return final_response
        except TypeError as e:
            return None


def wake_word_p():
    """

    :return:
    """
    import pvporcupine
    import pyaudio
    # Initialize Picovoice Porcupine with the custom wake word "Jarvis"
    porcupine = pvporcupine.create(
        access_key="6ze2tGeVN0YZHY2+5/nsR96nBYc2KV1kbMwAyYxNBdtKajMlhkkPDQ==",  # Replace with your Picovoice Access Key
        keyword_paths=["E:\\jarvis\\Client\\JARVIS\\Jarvis_en_windows_v3_0_0\\Jarvis_en_windows_v3_0_0.ppn"]  # Replace with the actual path to your "Jarvis" .ppn file
    )

    # Audio stream setup
    pa = pyaudio.PyAudio()
    audio_stream = pa.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length
    )

    print("Listening for 'Jarvis'...")

    try:
        while True:
            # Read audio data from the microphone
            pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
            pcm = [int.from_bytes(pcm[i:i + 2], byteorder='little', signed=True) for i in range(0, len(pcm), 2)]

            # Check if the wake word is detected
            keyword_index = porcupine.process(pcm)
            if keyword_index >= 0:
                print("Wake word 'Jarvis' detected!")
                return True
                break
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        # Cleanup resources
        audio_stream.close()
        pa.terminate()
        porcupine.delete()
    return False
def main():
    while True:
        try:
            if wake_word_p():
                user_input = recognize_speech()
                if "stop" in user_input:
                    break
                if user_input is not None:
                    final_response = jarvis_brain(user_input)
        except Exception as e:
            print(str(e))

class NLTKDownloadThread(threading.Thread):
    def run(self):
        import nltk
        # Perform the NLTK downloads in the thread to avoid blocking GUI
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('averaged_perceptron_tagger_eng')
        nltk.download('stopwords')
        nltk.download('punkt_tab')


def system(user_input):
    from open_interpreter import interpreter
    from env import HF_TOKEN, GROQ_APIKEY,NEXUSAI_API_KEY
    if NEXUSAI_API_KEY is not None or HF_TOKEN is not None or GROQ_APIKEY is not None:
        try:
            interpreter.llm.model = "huggingface/Qwen/Qwen2-VL-2B-Instruct"
            interpreter.llm.api_key = f"{HF_TOKEN}"
            interpreter.llm.supports_vision = True
            response = []
            for i in interpreter.chat(user_input, display=False):
                response.append(i.get("content"))
        except Exception as e:
            interpreter.llm.model = "groq/mixtral-8x7b-32768"
            interpreter.llm.api_key = f"{GROQ_APIKEY}"

            response = []
            for i in interpreter.chat(user_input, display=False):
                response.append(i.get('content'))
        finally:
            llm = LLM(model="jarvis:latest", api_key=f"{NEXUSAI_API_KEY}",
                      base_url=f"{NEXUSAI_API_BASE_URL}")
            response = [llm.call(
                messages=[{"role": "user", "content": user_input}]
            )
            ]
        final_response = summarize_results(user_input, response)
        return final_response


def gui():
    from gui.main import MainWindow
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

def autogen(port=8081,host="127.0.0.1"):
    from AutogenStudio.autogenstudio.web.app import app
    import uvicorn

    uvicorn.run(app,port=port,host=host)