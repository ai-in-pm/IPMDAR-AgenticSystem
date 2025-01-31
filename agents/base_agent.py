from abc import ABC, abstractmethod
import os
from dotenv import load_dotenv
import logging
from typing import List, Dict, Any

class BaseAgent(ABC):
    # RBAC Configuration
    USER_ROLES = {
        "admin": ["ingest", "analyze", "report", "configure"],
        "analyst": ["ingest", "analyze"],
        "viewer": ["report"]
    }

    def __init__(self, user_role: str = "analyst"):
        load_dotenv()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.user_role = user_role
        self.setup_api_keys()
        
    def setup_api_keys(self):
        """Setup API keys for various AI services"""
        self.openai_key = os.getenv('OPENAI_API_KEY')
        self.anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        self.groq_key = os.getenv('GROQ_API_KEY')
        self.google_key = os.getenv('GOOGLE_API_KEY')
        self.cohere_key = os.getenv('COHERE_API_KEY')
        self.emergence_key = os.getenv('EMERGENCEAI_API_KEY')
        
    def check_access(self, action: str) -> bool:
        """Ensures users can only perform actions based on their role."""
        return action in self.USER_ROLES.get(self.user_role, [])
        
    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the input data and return results"""
        pass
        
    def chain_of_thought(self, problem: str, process_steps: List[str], justification: str, output: Dict[str, Any]) -> Dict[str, Any]:
        """Document the chain of thought reasoning process"""
        cot = {
            "understanding": problem,
            "logical_process": process_steps,
            "justification": justification,
            "output": output
        }
        for step in process_steps:
            self.logger.info(f"CoT Reasoning: {step}")
        return cot
