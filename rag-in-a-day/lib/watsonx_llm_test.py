import pytest
from lib.model import Message, Role
from lib.watsonx_llm import WatsonxLLM

@pytest.fixture()
def llm():
    yield WatsonxLLM()

def test_ping_pong(llm):
    """Test if LLM replies with pong to ping"""
    msg = llm.complete_chat([
        Message.from_system("You are a helpful assistant. Be short and precise."),
        Message.from_user("If I say ping, you reply with")
    ])
    assert msg.role == Role.ASSISTANT
    assert msg.content.lower() in ["pong", "pong!"]

def test_simple_question_streaming(llm):
    """Test if LLM knows whether sky is blue"""
    msgs = [
        Message.from_system("You are a helpful assistant. Be short and precise."),
        Message.from_user("Is the sky blue? Yes or no")
    ]
    num_msg_chunks = 0
    for msg in llm.complete_chat(msgs, stream=True):
        assert msg.role == Role.ASSISTANT
        num_msg_chunks += 1
    assert num_msg_chunks > 1
    assert "yes" in msg.content.lower()
