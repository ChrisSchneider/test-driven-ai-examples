import time
import pytest
from unittest.mock import Mock, ANY
from streamlit.testing.v1.app_test import AppTest
from lib.model import Message, Role

@pytest.fixture
def mock_llm():
    """Fixture mocking WatsonxLLM"""
    mock_llm = Mock()
    mock_llm.complete_chat.return_value = Message.from_assistant("")
    return mock_llm


@pytest.fixture
def at(mock_llm):
    """Fixture that prepares the Streamlit app tests and mocks WatsonxLLM"""
    at = AppTest.from_file("app.py")
    at.session_state.llm = mock_llm
    at.run()
    return at


def test_app_starts(at):
    """Verify the app starts without exceptions"""
    assert not at.exception


def test_page_title(at):
    """Verify the app has the expected title"""
    assert len(at.subheader) == 1
    assert "Smart Assistant" in at.subheader[0].value 


def test_shows_welcome_message(at):
    """Verify welcome message from assistant is shown"""
    assert len(at.chat_message) == 1
    assert at.chat_message[0].name == "assistant"
    assert "Hello" in at.chat_message[0].markdown[0].value


def test_user_message_and_reply(at, mock_llm):
    """Verify users can submit a message and get a reply in a streaming fashion"""
    def complete_chat_mock(msgs, stream):
        yield Message.from_assistant("Hey")
        print(at._tree)
        yield Message.from_assistant("Hey there")
    mock_llm.complete_chat.side_effect = complete_chat_mock

    at.chat_input[0].set_value("Hi").run()

    assert 1 == 2
    mock_llm.complete_chat.assert_called_once()
    assert len(at.chat_message) == 3
    assert at.chat_message[1].name == "user"
    assert at.chat_message[1].markdown[0].value == "Hi"
    assert at.chat_message[2].name == "assistant"
    assert at.chat_message[2].markdown[0].value == "Hey there"


def test_handles_reply_error(at, mock_llm):
    """Verify an error message is shown if LLM returns an error"""
    mock_llm.complete_chat.side_effect = RuntimeError("Some error")

    at.chat_input[0].set_value("Hi").run()
    
    assert len(at.exception) == 1
    assert at.exception[0].message == "Some error"
