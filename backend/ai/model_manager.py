from typing import List, Dict

from ai.llm import LLM
from utils.logger import get_logger


class ModelManager:
    """
    Central model manager.

    This layer allows JARVIS to use different model
    implementations in the future.
    """

    def __init__(self):
        self.logger = get_logger(
            "JARVIS.ModelManager"
        )

        self.llm = LLM()

        self.logger.info(
            "Model Manager initialized"
        )

    def generate(
        self,
        instructions: str,
        messages: List[Dict[str, str]]
    ) -> str:

        return self.llm.generate(
            instructions=instructions,
            messages=messages
        )