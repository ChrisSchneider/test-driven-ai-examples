from typing import List
from lib.model import Role, Message

  
def generate_llama_prompt(messages: List[Message]) -> str:
    """
    Formats the dialog into a structure suitable for the Llama-2 chat model
    """
    prompt = "<s>[INST] <<SYS>>\n"

    # Add system messages
    system_messages = [msg.content for msg in messages if msg.role == Role.SYSTEM]
    if not system_messages:
        raise ValueError("Llama2 requires at least one system message")
    prompt += "\n\n".join(system_messages)

    prompt += "\n<</SYS>>\n\n"

    # Iterate over user and assistant messages
    last_role = None
    for msg in messages:
        if msg.role == Role.SYSTEM:
            continue
        if msg.role == Role.USER:
            if last_role == Role.ASSISTANT:
                prompt += "</s><s>[INST] "
            prompt += msg.content + " "
        if msg.role == Role.ASSISTANT:
            if not last_role or last_role == Role.USER:
                prompt += "[/INST] "
            prompt += msg.content + " "
        last_role = msg.role
    
    if last_role == Role.ASSISTANT:
        raise ValueError("Prompt cannot end with an assistant message")

    prompt += "[/INST]"

    return prompt


def parse_llama_response(text: str) -> Message:
    """
    Parses the response from Llama2 into a message
    """
    content = text.strip()
    if content.endswith("</s>"):
        content = content[:-4].strip()
    return Message.from_assistant(content)
