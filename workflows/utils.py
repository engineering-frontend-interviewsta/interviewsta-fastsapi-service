"""
Shared utilities for LangGraph workflows
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from typing import Optional
import os


def get_llm(google_api_key: str, temperature: float = 0.8, model: str = "models/gemini-2.0-flash"):
    """
    Get configured LLM instance
    
    Args:
        google_api_key: Google API key
        temperature: Model temperature (0-1)
        model: Model name
        
    Returns:
        ChatGoogleGenerativeAI: Configured LLM instance
    """
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=google_api_key
    )


def make_search_tool(tavily_api_key: str, max_results: int = 5):
    """
    Create Tavily search tool
    
    Args:
        tavily_api_key: Tavily API key
        max_results: Maximum number of search results
        
    Returns:
        Callable: Search function
    """
    search = TavilySearch(
        max_results=max_results,
        topic="general",
        tavily_api_key=tavily_api_key,
        include_answer=True
    )
    
    def get_google_search(query: str):
        """Call to perform google search online and get reliable results"""
        return search.invoke({"query": query})
    
    return get_google_search
