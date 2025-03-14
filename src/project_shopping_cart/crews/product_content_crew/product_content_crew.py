from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, WebsiteSearchTool,DirectoryReadTool,FileReadTool
from crewai import LLM
from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory
from dotenv import load_dotenv,find_dotenv
_:bool = load_dotenv(find_dotenv())
import os

from crewai.memory.storage import ltm_sqlite_storage as LTMSQLiteStorage, rag_storage as RAGStorage

# deep_seek_llm = LLM(
#     model="ollama/deepseek-r1:1.5b",
#     base_url="http://localhost:11434",
#     timeout=60
# )

gemini_llm = LLM(model='gemini/gemini-2.0-flash')

# Create tools
search_tool = SerperDevTool()
web_search_tool = WebsiteSearchTool()
docs_tool = DirectoryReadTool(directory='./blog-posts')
file_tool = FileReadTool()

storage_path = os.getenv("CREWAI_STORAGE_DIR", "./storage")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

google_embedder = {
     "provider": "google",
        "config": {
            "model": "models/text-embedding-004",
            "api_key": GEMINI_API_KEY,
        }
}


@CrewBase
class ProductContentCrew:
    """Product Content Writing Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def content_writer(self) -> Agent:
        return Agent(
            config=self.agents_config["content_writer"],
            tools=[docs_tool,file_tool],
            llm=gemini_llm,
            verbose=True,
        )
    @task
    def write_content(self) -> Task:
        return Task(
            config=self.tasks_config["write_content"],
        )
    

    @crew
    def crew(self) -> Crew:
        """Creates the Product Content Writing Crew"""

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            verbose=True,
            planning=True,
            planning_llm=gemini_llm,
            memory=True,

        )
