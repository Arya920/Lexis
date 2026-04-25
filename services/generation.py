from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

from services.prompts import PromptTemplates
from config.settings import GENERATION_MODEL_NAME
from services.query_logging import record_llm_call
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is not found in the env file.")

class AnswerGenerator:

    def __init__(self, model_name=GENERATION_MODEL_NAME):
        self.llm = init_chat_model(model_name)

    def generate_rag(self, query: str, context: str) -> str:
        messages = PromptTemplates.rag_prompt(context, query)
        response = self.llm.invoke(messages)
        record_llm_call(
            use_case="chat_rag_answer",
            output_text=response.content,
            response=response,
            model_name=GENERATION_MODEL_NAME,
        )
        return response.content

    def generate_web(self, query: str, context: str) -> str:
        messages = PromptTemplates.web_prompt(context, query)
        response = self.llm.invoke(messages)
        record_llm_call(
            use_case="chat_web_answer",
            output_text=response.content,
            response=response,
            model_name=GENERATION_MODEL_NAME,
        )
        return response.content

    def generate_direct(self, query: str) -> str:
        messages = PromptTemplates.direct_prompt(query)
        response = self.llm.invoke(messages)
        record_llm_call(
            use_case="chat_direct_answer",
            output_text=response.content,
            response=response,
            model_name=GENERATION_MODEL_NAME,
        )
        return response.content
