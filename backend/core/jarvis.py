from typing import Any, Dict, Optional

from core.brain import Brain
from core.conversation import Conversation
from core.context_engine import ContextEngine
from core.proactive import ProactiveEngine

from utils.logger import get_logger
from utils.helpers import clean_text


class Jarvis:

    def __init__(
        self,
        session_id: str = "default",
    ):

        self.logger = get_logger(
            "JARVIS"
        )

        self.brain = Brain()

        self.conversation = (
            Conversation(
                session_id=session_id
            )
        )

        self.context = ContextEngine()

        self.proactive = (
            ProactiveEngine()
        )

        self.logger.info(
            "JARVIS initialized"
        )

    # =========================================================
    # Normal conversation
    # =========================================================

    def receive(
        self,
        user_input: str,
    ) -> dict:

        user_input = clean_text(
            user_input
        )

        if not user_input:

            return {
                "success": False,
                "message": "No input received.",
            }

        self.logger.info(
            "User input received"
        )

        self.conversation.add_message(
            role="user",
            content=user_input,
        )

        context = (
            self.context.snapshot()
        )

        result = self.brain.think(
            user_input=user_input,
            context=context,
            history=(
                self.conversation
                .get_history()
            ),
        )

        if result.get("success"):

            response = result.get(
                "response",
                "",
            )

            if response:

                self.conversation.add_message(
                    role="assistant",
                    content=response,
                )

        return result

    # =========================================================
    # Proactive reasoning
    # =========================================================

    def evaluate_proactive(
        self,
    ) -> Dict[str, Any]:
        """
        Evaluate whether JARVIS should proactively
        interact with the user.

        This does not generate speech.

        It only decides whether interaction is appropriate.
        """

        context = (
            self.context.snapshot()
        )

        history = (
            self.conversation
            .get_history()
        )

        decision = (
            self.proactive.evaluate(
                context=context,
                history=history,
            )
        )

        return decision

    # =========================================================
    # Proactive response generation
    # =========================================================

    def generate_proactive_response(
        self,
        decision: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate a natural JARVIS response after the
        proactive engine decides that speaking is appropriate.

        This uses the same local LLM as normal conversation.
        """

        if not decision.get(
            "should_speak"
        ):

            return {
                "success": False,
                "spoken": False,
                "reason": "Proactive engine decided to wait.",
                "decision": decision,
            }

        reason = decision.get(
            "reason",
            "context_change",
        )

        context = (
            self.context.snapshot()
        )

        history = (
            self.conversation
            .get_history()
        )

        proactive_instruction = f"""
You are JARVIS, a human-like AI companion.

You have decided that it may be appropriate to
proactively speak to the user.

Reason:
{reason}

Current context:
{context}

Recent conversation:
{history[-8:]}

Speak naturally, like an intelligent person who is
present with the user.

Do not sound like a notification.
Do not announce internal system decisions.
Do not mention the word "proactive".
Do not give a command.
Do not use a fixed template.

Only speak if there is genuinely something useful,
interesting, relevant, or natural to say.

Keep the response concise.
Usually one or two sentences are enough.
"""

        try:

            response = (
                self.brain.model_manager.generate(
                    instructions=(
                        proactive_instruction
                    ),
                    messages=[],
                )
            )

        except Exception as exc:

            self.logger.error(
                "Proactive response generation failed: "
                f"{exc}"
            )

            return {
                "success": False,
                "spoken": False,
                "reason": "generation_failed",
                "error": str(exc),
                "decision": decision,
            }

        if not response:

            return {
                "success": False,
                "spoken": False,
                "reason": "empty_response",
                "decision": decision,
            }

        self.conversation.add_message(
            role="assistant",
            content=response,
        )

        return {
            "success": True,
            "spoken": True,
            "response": response,
            "reason": reason,
            "priority": decision.get(
                "priority"
            ),
            "decision": decision,
        }

    # =========================================================
    # Context update
    # =========================================================

    def update_context(
        self,
        key: str,
        value: Any,
    ) -> Dict[str, Any]:

        self.context.update(
            key,
            value,
        )

        return {
            "success": True,
            "context": (
                self.context.snapshot()
            ),
        }

    # =========================================================
    # Full proactive cycle
    # =========================================================

    def proactive_cycle(
        self,
    ) -> Dict[str, Any]:
        """
        Run the complete proactive reasoning cycle.

        Observe
            ↓
        Evaluate
            ↓
        Decide
            ↓
        Optionally speak
        """

        decision = (
            self.evaluate_proactive()
        )

        if not decision.get(
            "should_speak"
        ):

            return {
                "success": True,
                "spoken": False,
                "decision": decision,
            }

        return (
            self.generate_proactive_response(
                decision
            )
        )

    # =========================================================
    # Status
    # =========================================================

    def status(self) -> dict:

        return {
            "name": "JARVIS",
            "status": "online",
            "session_id": (
                self.conversation.session_id
            ),
            "messages": (
                self.conversation
                .message_count()
            ),
            "proactive": (
                self.proactive.status()
            ),
        }

    # =========================================================
    # Reset conversation
    # =========================================================

    def reset_conversation(
        self,
    ):

        self.conversation.clear()

        self.proactive.reset()

        self.logger.info(
            "JARVIS conversation reset"
        )