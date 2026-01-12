import sys
from pathlib import Path

src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv  # api key
from langchain_core.prompts import PromptTemplate

load_dotenv()


class GenerateService:
    '''
    Generate answer based on Open AI model
    '''

    def __init__(self, model: str = 'gpt-5'):
        self.model = model
        self.llm = ChatOpenAI(model=model, temperature=0)

    def generate(self, prompt: PromptTemplate, context: str, question: str):
        '''Generate answer from prompt template'''
        formatted_prompt = prompt.format(context=context, question=question)
        response = self.llm.invoke(formatted_prompt)
        return response.content
