from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, WebsiteSearchTool,DirectoryReadTool,FileReadTool
from crewai import LLM
from dotenv import load_dotenv,find_dotenv
_:bool = load_dotenv(find_dotenv())

from crewai.memory.storage import ltm_sqlite_storage as LTMSQLiteStorage, rag_storage as RAGStorage

deep_seek_llm = LLM(
    model="ollama/deepseek-r1:1.5b",
    base_url="http://localhost:11434",
    timeout=60
)

gemini_llm = LLM(model='gemini/gemini-2.0-flash')

# Create tools
search_tool = SerperDevTool()
web_search_tool = WebsiteSearchTool()

@CrewBase
class ShoppingCrew:
    """Shopping Crew"""
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher"],
            tools=[web_search_tool,search_tool],
            llm=gemini_llm,
            verbose=True
        )
    

    @task
    def find_category(self) -> Task:
        return Task(
            config=self.tasks_config["find_category"],
        )


    @task
    def trending_product(self) -> Task:
        return Task(
            config=self.tasks_config["trending_product"],
        )
    
    @task
    def trending_product_price(self) -> Task:
        return Task(
            config=self.tasks_config["trending_product_price"],
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the Product Crew"""

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            verbose=True,
            process=Process.sequential
        )
