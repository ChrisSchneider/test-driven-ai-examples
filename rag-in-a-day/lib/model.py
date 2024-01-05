from enum import StrEnum
from pydantic import BaseModel


class Role(StrEnum):
    ASSISTANT = "assistant"
    USER = "user"
    SYSTEM = "system"


class Message(BaseModel):
    role: Role
    content: str

    @classmethod
    def from_assistant(cls, content: str) -> "Message":
        """Create a message from the assistant role."""
        return cls(role=Role.ASSISTANT, content=content)

    @classmethod
    def from_user(cls, content: str) -> "Message":
        """Create a message from the user role."""
        return cls(role=Role.USER, content=content)

    @classmethod
    def from_system(cls, content: str) -> "Message":
        """Create a message from the system role."""
        return cls(role=Role.SYSTEM, content=content)
