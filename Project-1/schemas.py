"""
Docstring for Project-1.schemas

This takes the output from the LLM and formats it in a good way!
"""

from typing import List

from pydantic import BaseModel, Field


class Source(BaseModel):
    """
    Schema of the Source used by the agent
    """

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """
    Schema for agent response with answer and sources
    """

    answer: str = Field(description="The answer from the LLM")
    sources: List[Source] = Field(
        desription="The list of sources used by the LLM", default_factory=list
    )
