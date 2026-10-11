from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class Message:
    """
    Represents one message in a conversation.
    """

    role: str
    content: str
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )


class Conversation:
    """
    Stores the conversation history for one JARVIS session.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: List[Message] = []

    def add_message(
        self,
        role: str,
        content: str
    ) -> Message:

        message = Message(
            role=role,
            content=content
        )

        self.messages.append(message)

        return message

    def get_history(self) -> List[Dict[str, str]]:
        """
        Return conversation history in a format
        that the Brain can understand.
        """

        return [
            {
                "role": message.role,
                "content": message.content
            }
            for message in self.messages
        ]

    def last_message(self) -> Optional[Message]:
        """
        Return the latest message.
        """

        if not self.messages:
            return None

        return self.messages[-1]

    def clear(self) -> None:
        """
        Clear the current conversation.
        """

        self.messages.clear()

    def message_count(self) -> int:
        """
        Return the number of messages.
        """

        return len(self.messages)