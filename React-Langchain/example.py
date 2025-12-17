from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate

# tool example
@tool
def multiply(a: int, b: int):
    """Multiply two numbers."""
    return a * b

llm = ChatOpenAI(model="gpt-4o", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Use ReAct reasoning to solve tasks with tools."),
    ("human", "{input}")
])

agent = create_react_agent(
    llm=llm,
    tools=[multiply],
    prompt=prompt
)

executor = AgentExecutor(agent=agent, tools=[multiply], verbose=True)

response = executor.invoke({"input": "What is 12 × 8?"})
print(response)
