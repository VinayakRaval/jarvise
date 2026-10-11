from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional


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
    Represents one complete conversational session.

    A conversation contains:
    - session ID
    - messages
    - timestamps
    - conversation history
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

        return [
            {
                "role": message.role,
                "content": message.content
            }
            for message in self.messages
        ]


    def last_message(self) -> Optional[Message]:

        if not self.messages:
            return None

        return self.messages[-1]


    def clear(self):

        self.messages.clear()


    def message_count(self) -> int:

        return len(self.messages)