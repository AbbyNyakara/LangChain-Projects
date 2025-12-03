PROMPT_WITH_FORMAT_INSTRUCTIONS = """
Answer the following questions best you can. You have access to the following tools {tools}

Use the following format: 

Question: The question you must answer
thought: You should always think about what to do
Action: the action you should take. Should be one of the [{tool_names}]
Action_input: the input to the action
Observation: the result of the action
.... this (Thought/action/Action_input/Observation can repeat N times)

Thought:I now know the final answer
Final Answer : The final answer to the original input question, formatted according to the {format_instructions}

Begin

Question: {input}
Thought: {agent_scratchpad}
"""
