from langchain_openai import ChatOpenAI
from app.config import OPENAI_API_KEY

# Initialize main async LLM client once
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=OPENAI_API_KEY)
