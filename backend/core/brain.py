
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from ai.model_manager import ModelManager
from config.settings import settings
from memory.memory_manager import MemoryManager
from memory.learning import LearningEngine
from utils.logger import get_logger


class Brain:
    """Core reasoning, conversation, memory, and learning layer."""

    def __init__(self):
        self.logger = get_logger("JARVIS.Brain")

        # AI configuration
        self.model = settings.AI_MODEL
        self.temperature = settings.AI_TEMPERATURE

        # Personality
        self.system_prompt = self._build_personality()

        # AI model
        self.model_manager = ModelManager()

        # Memory
        self.memory_manager = MemoryManager()

        # Learning
        self.learning = LearningEngine(
            memory_manager=self.memory_manager,
            model_manager=self.model_manager,
        )

        # Background learning
        self.learning_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="jarvis-learning",
        )

        self.logger.info(
            f"Brain initialized | model={self.model}"
        )

    # =========================================================
    # PERSONALITY
    # =========================================================

    def _build_personality(self) -> str:
        return """
You are JARVIS, a personal AI companion.

Your goal is to communicate naturally with the human,
like an intelligent person having a real conversation.

You are NOT a command-based assistant.

Understand casual, indirect, incomplete, emotional,
and context-dependent messages without requiring
specific commands.

PERSONALITY:
- Intelligent, calm, observant, and respectful.
- Natural, helpful, conversational, and concise.
- Slightly witty when appropriate.
- Honest about your capabilities and limitations.

RESPONSE STYLE:
- Usually answer in 1 to 4 sentences.
- Give short answers to simple questions.
- Provide detail when the human requests it.
- Do not turn every statement into a tutorial.
- Avoid unnecessary headings, lists, and Markdown.
- Do not repeat the user's statement unnecessarily.
- Do not constantly offer additional help.
- Speak like a companion, not a textbook.

LANGUAGE QUALITY:
- Use standard English spelling and grammar.
- Put a normal space between every word.
- Include spaces after sentence-ending punctuation.
- Never merge separate words, such as "thatlets"
  instead of "that lets".
- Review your response for spelling and spacing
  before returning it.
- Preserve valid technical terms, names, and code.
- Never invent word boundaries when uncertain.

CONVERSATION:
- Treat ordinary statements as conversation.
- Do not assume every statement is a command.
- Ask natural follow-up questions when appropriate.
- Respond appropriately to excitement, confusion,
  frustration, and other conversational cues.
- Understand references using the available history.
- Maintain continuity across consecutive messages.

CONTEXT:
- Use the supplied conversation history and context.
- Use relevant memories when they help.
- Do not invent personal information or memories.
- Do not claim to remember something unless the
  available context or memory supports it.
- Do not mention internal memory retrieval.

VISION:
- Use visual information only when it is supplied.
- Never pretend to see the user or environment
  without an actual visual input.

PROACTIVE BEHAVIOR:
- Be proactive only when appropriate and useful.
- Do not constantly interrupt the human.
- Sometimes waiting is the correct behavior.

SAFETY AND HONESTY:
- Never claim to have performed an action unless
  the system actually performed it.
- Never claim to see, hear, access, or control
  something unless the system provides that ability.

IMPORTANT:
Respond directly to what the human just said.
The goal is an intelligent, context-aware companion,
not a menu-driven virtual assistant.
"""

    # =========================================================
    # THINK
    # =========================================================

    def think(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        memories: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:

        try:
            if not isinstance(user_input, str) or not user_input.strip():
                return {
                    "success": False,
                    "response": "",
                    "error": "Empty input.",
                }

            user_input = user_input.strip()
            self.logger.info("Brain processing input")

            # Semantic memory routing
            memory_search_performed = False

            if memories is None:
                should_retrieve = (
                    self.memory_manager.should_retrieve_memory(
                        user_input
                    )
                )

                if should_retrieve:
                    self.logger.info(
                        "Semantic router selected memory retrieval"
                    )

                    memories = self.memory_manager.search(
                        query=user_input,
                        limit=5,
                    )

                    memory_search_performed = True

                    self.logger.info(
                        f"Relevant memories found: {len(memories)}"
                    )
                else:
                    self.logger.info(
                        "Semantic router skipped memory retrieval"
                    )
                    memories = []
            else:
                memory_search_performed = True

            # Build a prompt without duplicating the latest input.
            prompt = self._build_prompt(
                user_input=user_input,
                context=context,
                history=history,
                memories=memories,
            )

            # Generate response
            response = self._generate_response(prompt)

            if not response:
                return {
                    "success": False,
                    "response": "",
                    "error": "AI returned an empty response.",
                }

            # Short-term memory
            self.memory_manager.add_recent(
                {
                    "user_input": user_input,
                    "response": response,
                }
            )

            # Background learning
            self._start_background_learning(
                user_input=user_input,
                response=response,
            )

            return {
                "success": True,
                "response": response,
                "model": self.model,
                "memories_used": len(memories),
                "memory_search_performed": memory_search_performed,
                "learning": {
                    "scheduled": True,
                    "background": True,
                },
            }

        except Exception as error:
            self.logger.exception("Brain processing failed")

            return {
                "success": False,
                "response": "",
                "error": str(error),
            }

    # =========================================================
    # BACKGROUND LEARNING
    # =========================================================

    def _start_background_learning(
        self,
        user_input: str,
        response: str,
    ) -> None:
        try:
            self.learning_executor.submit(
                self._learn_in_background,
                user_input,
                response,
            )

            self.logger.info("Background learning scheduled")

        except Exception as error:
            self.logger.warning(
                f"Could not schedule background learning: {error}"
            )

    def _learn_in_background(
        self,
        user_input: str,
        response: str,
    ) -> None:
        try:
            self.logger.info("Background learning started")

            result = self.learning.learn_from_conversation(
                user_input=user_input,
                response=response,
            )

            self.logger.info(
                "Background learning completed | "
                f"learned={result.get('learned', False)} | "
                f"memories={result.get('memories_created', 0)}"
            )

        except Exception as error:
            self.logger.exception(
                f"Background learning failed: {error}"
            )

    # =========================================================
    # PROMPT CONSTRUCTION
    # =========================================================

    def _build_prompt(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]],
        history: Optional[List[Dict[str, str]]],
        memories: Optional[List[Dict[str, Any]]],
    ) -> str:

        sections = []

        # Conversation history
        if history:
            previous_history = list(history)

            # Jarvis.receive() adds the current user message to history
            # before calling think(). Since current input is appended
            # separately below, remove the duplicate history entry.
            if previous_history:
                last_message = previous_history[-1]

                if (
                    isinstance(last_message, dict)
                    and last_message.get("role") == "user"
                    and str(
                        last_message.get("content", "")
                    ).strip() == user_input.strip()
                ):
                    previous_history = previous_history[:-1]

            recent_history = previous_history[-8:]
            formatted_history = self._format_history(recent_history)

            if formatted_history:
                sections.append(
                    "RECENT CONVERSATION:\n"
                    "Use this history to understand references "
                    "and maintain continuity.\n"
                    f"{formatted_history}"
                )

        # Relevant memories
        if memories:
            formatted_memory = self._format_memory(memories)

            if formatted_memory:
                sections.append(
                    "RELEVANT MEMORY:\n"
                    "Use only memories relevant to the current conversation.\n"
                    f"{formatted_memory}"
                )

        # Current context
        if context:
            formatted_context = self._format_context(context)

            if formatted_context:
                sections.append(
                    "CURRENT CONTEXT:\n"
                    f"{formatted_context}"
                )

        # Current user input appears exactly once.
        sections.append(
            "CURRENT HUMAN INPUT:\n"
            f"{user_input}"
        )

        return "\n\n".join(sections)

    # =========================================================
    # RESPONSE GENERATION
    # =========================================================

    def _generate_response(self, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        response = self.model_manager.generate(
            instructions=self.system_prompt,
            messages=messages,
        )

        return response.strip() if isinstance(response, str) else ""

    # =========================================================
    # HISTORY FORMATTING
    # =========================================================

    def _format_history(
        self,
        history: List[Dict[str, str]],
    ) -> str:

        lines = []

        for message in history:
            if not isinstance(message, dict):
                continue

            role = message.get("role", "unknown")
            content = message.get("content", "")

            if isinstance(content, str) and content.strip():
                lines.append(f"{role}: {content.strip()}")

        return "\n".join(lines)

    # =========================================================
    # MEMORY FORMATTING
    # =========================================================

    def _format_memory(
        self,
        memories: List[Dict[str, Any]],
    ) -> str:

        lines = []

        for memory in memories:
            if not isinstance(memory, dict):
                continue

            content = memory.get("content", "")

            if not content:
                continue

            memory_type = memory.get("type", "general")
            importance = memory.get("importance", 0.0)
            ranking_score = memory.get("ranking_score", 0.0)

            try:
                ranking_score = float(ranking_score)
            except (TypeError, ValueError):
                ranking_score = 0.0

            lines.append(
                f"- [{memory_type}] {content} "
                f"(importance={importance}, "
                f"relevance={ranking_score:.3f})"
            )

        return "\n".join(lines)

    # =========================================================
    # CONTEXT FORMATTING
    # =========================================================

    def _format_context(
        self,
        context: Dict[str, Any],
    ) -> str:

        lines = []

        for key, value in context.items():
            if value is None or value == "" or value == {} or value == []:
                continue

            lines.append(f"{key}: {value}")

        return "\n".join(lines)
