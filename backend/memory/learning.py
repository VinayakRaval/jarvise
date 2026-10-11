import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ai.model_manager import ModelManager
from memory.memory_manager import MemoryManager
from utils.logger import get_logger


class LearningEngine:
    """
    AI-based continuous learning system for JARVIS.

    The local AI model decides whether a conversation contains
    information worth remembering.

    This is NOT neural-network retraining.

    Instead:

        Conversation
             ↓
        Fast memory gate
             ↓
        Is this potentially important?
             ↓
        Local AI analysis
             ↓
        Memory decision
             ↓
        Structured memory
             ↓
        Long-term semantic storage

    The fast memory gate prevents unnecessary AI calls for
    ordinary conversation such as greetings and small talk.
    """

    def __init__(
        self,
        memory_manager: Optional[MemoryManager] = None,
        model_manager: Optional[ModelManager] = None,
    ):
        self.logger = get_logger("JARVIS.Learning")

        # Reuse the existing memory system.
        self.memory = (
            memory_manager
            if memory_manager is not None
            else MemoryManager()
        )

        # Reuse the same local AI model used by Brain.
        self.model_manager = (
            model_manager
            if model_manager is not None
            else ModelManager()
        )

        self.logger.info(
            "AI Learning Engine initialized"
        )

    # ============================================================
    # FAST MEMORY GATE
    # ============================================================

    def _should_learn(self, user_input: str) -> bool:
        """
        Quickly determine whether a user message is likely
        to contain information useful for long-term memory.

        This is intentionally lightweight and does NOT call
        the AI model.

        Examples that normally return False:

            Hello
            Hi
            Thanks
            Okay
            Bye

        Examples that may return True:

            I am building a JARVIS AI project.

            I prefer Python for backend development.

            My goal is to become a DevOps engineer.

            I am currently studying BE Computer Science.

        The AI memory extractor will make the final decision.
        """

        text = user_input.lower().strip()

        if not text:
            return False

        # --------------------------------------------------------
        # Very short casual conversation
        # --------------------------------------------------------

        casual_phrases = {
            "hi",
            "hello",
            "hey",
            "hey jarvis",
            "hi jarvis",
            "hello jarvis",
            "good morning",
            "good afternoon",
            "good evening",
            "good night",
            "thanks",
            "thank you",
            "thank u",
            "ok",
            "okay",
            "yes",
            "no",
            "sure",
            "great",
            "good",
            "fine",
            "cool",
            "nice",
            "bye",
            "goodbye",
        }

        if text in casual_phrases:
            return False

        # --------------------------------------------------------
        # Very short messages are usually not long-term memories.
        #
        # We still allow longer messages because they may contain
        # preferences, goals, projects, plans, etc.
        # --------------------------------------------------------

        if len(text) < 12:
            return False

        # --------------------------------------------------------
        # Signals that the human may be providing information
        # useful in future conversations.
        # --------------------------------------------------------

        memory_signals = [
            # Identity / personal context
            "i am ",
            "i'm ",
            "my name",
            "i live",
            "i stay",
            "i study",
            "i'm studying",
            "i am studying",

            # Preferences
            "i like ",
            "i love ",
            "i prefer ",
            "i don't like ",
            "i dislike ",
            "my favorite",
            "my favourite",

            # Goals
            "i want ",
            "i need ",
            "my goal",
            "my plan",
            "i plan to",
            "i hope to",
            "i'm trying to",
            "i am trying to",

            # Projects / work
            "my project",
            "i'm working on",
            "i am working on",
            "i'm building",
            "i am building",
            "i'm developing",
            "i am developing",
            "i created",
            "i built",

            # Skills / learning
            "i know ",
            "i learned ",
            "i am learning",
            "i'm learning",
            "i completed",
            "i finished",
            "i use ",
            "i work with",

            # Routines / recurring information
            "i usually",
            "i always",
            "i never",
            "every day",
            "every morning",
            "every evening",

            # Explicit memory requests
            "remember ",
            "don't forget",
            "do not forget",
            "from now on",
            "in the future",

            # Decisions / important context
            "i decided",
            "i have decided",
            "i chose",
            "i will use",
            "i'm going to",
            "i am going to",
        ]

        for signal in memory_signals:
            if signal in text:
                return True

        return False

    # ============================================================
    # MAIN LEARNING FUNCTION
    # ============================================================

    def learn_from_conversation(
        self,
        user_input: str,
        response: str,
    ) -> Dict[str, Any]:
        """
        Analyze a conversation using the local AI model.

        The process is:

            1. Validate input
            2. Run fast memory gate
            3. Ask AI to extract useful memories
            4. Validate extracted memories
            5. Store valuable memories
        """

        # --------------------------------------------------------
        # Validate user input
        # --------------------------------------------------------

        if not user_input or not user_input.strip():
            return {
                "learned": False,
                "memories_created": 0,
                "reason": "Empty user input",
            }

        # --------------------------------------------------------
        # Validate assistant response
        # --------------------------------------------------------

        if not response or not response.strip():
            return {
                "learned": False,
                "memories_created": 0,
                "reason": "Empty assistant response",
            }

        # --------------------------------------------------------
        # FAST MEMORY GATE
        #
        # This prevents a second Qwen3 call for ordinary
        # conversation.
        # --------------------------------------------------------

        if not self._should_learn(user_input):
            self.logger.info(
                "Memory gate: no learning required"
            )

            return {
                "learned": False,
                "memories_created": 0,
                "reason": (
                    "Conversation not likely to contain "
                    "long-term information"
                ),
            }

        try:

            # ----------------------------------------------------
            # Ask local AI to extract useful memories
            # ----------------------------------------------------

            self.logger.info(
                "Memory gate passed | running AI memory extraction"
            )

            extracted = self._extract_memories_with_ai(
                user_input=user_input,
                response=response,
            )

            # ----------------------------------------------------
            # Nothing useful detected
            # ----------------------------------------------------

            if not extracted:
                self.logger.info(
                    "No important memory detected"
                )

                return {
                    "learned": False,
                    "memories_created": 0,
                    "reason": "No important memory detected",
                }

            # ----------------------------------------------------
            # Store extracted memories
            # ----------------------------------------------------

            stored_memories: List[Dict[str, Any]] = []

            for memory in extracted:

                validated = self._validate_memory(
                    memory
                )

                if validated is None:
                    continue

                # Avoid storing extremely low-value memories.
                if validated["importance"] < 0.50:
                    self.logger.info(
                        "Memory rejected because importance "
                        f"is too low: {validated['importance']}"
                    )
                    continue

                result = self.memory.remember(
                    content=validated["content"],
                    memory_type=validated["type"],
                    importance=validated["importance"],
                    metadata={
                        "source": "ai_memory_extraction",
                        "created_at": datetime.now(
                            timezone.utc
                        ).isoformat(),
                    },
                )

                stored_memories.append(result)

            # ----------------------------------------------------
            # Log result
            # ----------------------------------------------------

            self.logger.info(
                "AI learning completed | "
                f"memories_created={len(stored_memories)}"
            )

            return {
                "learned": len(stored_memories) > 0,
                "memories_created": len(stored_memories),
                "memories": stored_memories,
            }

        except Exception as error:

            self.logger.exception(
                f"AI learning failed: {error}"
            )

            # Learning failure should NOT destroy the
            # normal conversation.

            return {
                "learned": False,
                "memories_created": 0,
                "error": str(error),
            }

    # ============================================================
    # AI MEMORY EXTRACTION
    # ============================================================

    def _extract_memories_with_ai(
        self,
        user_input: str,
        response: str,
    ) -> List[Dict[str, Any]]:
        """
        Ask the local Qwen model to determine what should
        become long-term memory.
        """

        instructions = """
You are the memory system of a personal AI companion called JARVIS.

Your job is to identify information from a conversation that would
genuinely help JARVIS understand the human better in future
conversations.

Do NOT remember everything.

Only extract information that is useful beyond the current
conversation.

Potential memory categories include:

- preference
- goal
- project
- skill
- important_context
- personal_context
- decision
- experience
- relationship
- routine
- general

IMPORTANT RULES:

1. Only remember information about the human.
2. Do not remember information invented by JARVIS.
3. Do not store temporary questions.
4. Do not store ordinary greetings.
5. Do not store casual small talk.
6. Do not store obvious information.
7. Do not store information that is useful only for the current reply.
8. Do not create memories from speculation.
9. Do not infer facts that the human did not provide.
10. Do not store sensitive information unless the human clearly
    and intentionally provides it as something important to remember.
11. Keep each memory short and factual.
12. Prefer one strong memory over several weak memories.
13. Do not duplicate the same information.
14. If the human explicitly asks JARVIS to remember something,
    treat that as strong evidence that it should be remembered.

For each memory return:

{
    "remember": true,
    "content": "short factual memory",
    "type": "category",
    "importance": 0.0
}

Importance must be between 0.0 and 1.0.

Importance guide:

0.0 - 0.3 = very low value
0.4 - 0.5 = somewhat useful
0.6 - 0.7 = useful
0.8 - 0.9 = very important
1.0 = extremely important

If nothing is worth remembering, return:

{
    "remember": false,
    "memories": []
}

If there are multiple useful memories, return:

{
    "remember": true,
    "memories": [
        {
            "content": "...",
            "type": "...",
            "importance": 0.0
        }
    ]
}

Return ONLY valid JSON.

Do not use Markdown.
Do not use code fences.
Do not explain your answer.
"""

        conversation = f"""
HUMAN:
{user_input}

JARVIS:
{response}
"""

        messages = [
            {
                "role": "user",
                "content": conversation,
            }
        ]

        raw_result = self.model_manager.generate(
            instructions=instructions,
            messages=messages,
        )

        return self._parse_ai_result(
            raw_result
        )

    # ============================================================
    # PARSE AI RESULT
    # ============================================================

    def _parse_ai_result(
        self,
        raw_result: str,
    ) -> List[Dict[str, Any]]:
        """
        Convert model output into Python dictionaries.

        Handles:

        - JSON object
        - JSON array
        - Markdown code fences
        - Extra whitespace
        """

        if not raw_result:
            return []

        text = raw_result.strip()

        # --------------------------------------------------------
        # Remove accidental Markdown code fences.
        # --------------------------------------------------------

        text = re.sub(
            r"^```(?:json)?",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"```$",
            "",
            text,
        )

        text = text.strip()

        # --------------------------------------------------------
        # Try direct JSON parsing first.
        # --------------------------------------------------------

        try:
            data = json.loads(text)

        except json.JSONDecodeError:

            # ----------------------------------------------------
            # Sometimes local models put extra text around JSON.
            # Try to extract the first JSON object or array.
            # ----------------------------------------------------

            json_object = re.search(
                r"\{.*\}",
                text,
                flags=re.DOTALL,
            )

            json_array = re.search(
                r"\[.*\]",
                text,
                flags=re.DOTALL,
            )

            candidate = None

            if json_object:
                candidate = json_object.group(0)

            elif json_array:
                candidate = json_array.group(0)

            if not candidate:
                self.logger.warning(
                    "AI memory output was not valid JSON"
                )
                return []

            try:
                data = json.loads(candidate)

            except json.JSONDecodeError:
                self.logger.warning(
                    "AI memory output could not be parsed"
                )
                return []

        # --------------------------------------------------------
        # Format:
        #
        # {
        #     "remember": true,
        #     "memories": [...]
        # }
        # --------------------------------------------------------

        if isinstance(data, dict):

            if data.get("remember") is False:
                return []

            memories = data.get(
                "memories",
                [],
            )

            # ----------------------------------------------------
            # Support a single-memory response:
            #
            # {
            #     "remember": true,
            #     "content": "...",
            #     "type": "...",
            #     "importance": 0.8
            # }
            # ----------------------------------------------------

            if (
                not memories
                and data.get("content")
            ):
                memories = [data]

            if isinstance(memories, list):
                return [
                    item
                    for item in memories
                    if isinstance(item, dict)
                ]

            return []

        # --------------------------------------------------------
        # Format:
        #
        # [
        #     {...},
        #     {...}
        # ]
        # --------------------------------------------------------

        if isinstance(data, list):

            return [
                item
                for item in data
                if isinstance(item, dict)
            ]

        return []

    # ============================================================
    # VALIDATE MEMORY
    # ============================================================

    def _validate_memory(
        self,
        memory: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Validate and normalize an AI-generated memory.
        """

        if not isinstance(memory, dict):
            return None

        content = memory.get(
            "content",
            "",
        )

        memory_type = memory.get(
            "type",
            "general",
        )

        importance = memory.get(
            "importance",
            0.5,
        )

        remember = memory.get(
            "remember",
            True,
        )

        # --------------------------------------------------------
        # Basic validation
        # --------------------------------------------------------

        if not remember:
            return None

        if not isinstance(content, str):
            return None

        content = content.strip()

        if not content:
            return None

        # --------------------------------------------------------
        # Prevent extremely large memories.
        # --------------------------------------------------------

        if len(content) > 1000:
            content = content[:1000].rstrip()

        # --------------------------------------------------------
        # Normalize memory type
        # --------------------------------------------------------

        allowed_types = {
            "preference",
            "goal",
            "project",
            "skill",
            "important_context",
            "personal_context",
            "decision",
            "experience",
            "relationship",
            "routine",
            "general",
        }

        if not isinstance(memory_type, str):
            memory_type = "general"

        memory_type = memory_type.strip().lower()

        if memory_type not in allowed_types:
            memory_type = "general"

        # --------------------------------------------------------
        # Normalize importance
        # --------------------------------------------------------

        try:
            importance = float(
                importance
            )

        except (
            TypeError,
            ValueError,
        ):
            importance = 0.5

        importance = max(
            0.0,
            min(
                1.0,
                importance,
            ),
        )

        # --------------------------------------------------------
        # Final normalized memory
        # --------------------------------------------------------

        return {
            "content": content,
            "type": memory_type,
            "importance": importance,
        }

    # ============================================================
    # STATUS
    # ============================================================

    def status(self) -> Dict[str, Any]:
        """
        Return learning system status.
        """

        return {
            "learning_enabled": True,
            "method": "local_ai_memory_extraction",
            "memory_gate": True,
            "memory_status": self.memory.status(),
        }