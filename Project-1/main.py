from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool
from langchain.agents import create_agent
from schemas import AgentResponse  

load_dotenv()

@tool
def search_linkedin_jobs(query: str) -> str:
    """Search LinkedIn for AI/NLP engineer job postings."""
    return TavilySearchResults(max_results=3).invoke(query)

tools = [TavilySearchResults(max_results=3)]  # Or just [TavilySearchResults(max_results=3)]
llm = ChatOpenAI(model="gpt-4o", temperature=0)

agent = create_agent(
    model=llm,
    tools=tools
)

def main():
    result = agent.invoke({
        "messages": [{"role": "user", 
                      "content": "Search for 3 job postings for AI engineer"
                      "/NLP engineer using LinkedIn"}]
    })
    
    print(result)

if __name__ == "__main__":
    main()
