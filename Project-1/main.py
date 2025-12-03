from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchainhub import hub
from langchain_tavily import TavilySearch
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent

load_dotenv()

tools = [TavilySearch()]
llm = ChatOpenAI(model='gpt-4', temperature=0)

react_prompt = hub.pull('hwchase17/react')

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt
)

agent_executor=AgentExecutor(agent=agent, tool=tools, verbose=True)
chain = agent_executor


def main():
    print("Hello from project-1!")


if __name__ == "__main__":
    main()
