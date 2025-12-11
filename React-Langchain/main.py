from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import render_text_description
from langchain_openai import ChatOpenAI

load_dotenv()
'''
LangChain’s @tool decorator uses the function’s docstring as the tool’s
 description so the LLM knows what the tool does and when to call it. - the docstring is required
'''


@tool
def get_text_length(text: str) -> int:
    '''
    Returns the length of the given text
    '''
    return len(text)


if __name__ == "__main__":
    tools = [get_text_length]

    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:
    """

    # The llm only acceps strings. The function render_text_description will be converted to a string
    prompt = PromptTemplate.from_template(
        template=template).partial(tools=render_text_description(tools),
                                   tool_names=", ".join([t.name for t in tools]))

    llm = ChatOpenAI(temperature=0, stop=["\nObservation"])
    # lcel - pipe operator takesthe output from the left into the right
    agent = {"input": lambda x: x["input"]} | prompt | llm

    response = agent.invoke(
        {"input": "What is the number of characters in the word 'DOG' "})
    print(response)
