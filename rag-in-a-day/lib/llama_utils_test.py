import pytest
from lib.model import Role, Message
from lib.llama_utils import generate_llama_prompt, parse_llama_response

def test_prompt_for_single_turn_converstation():
    messages = [
        Message.from_system("You are an assistant."),
        Message.from_user("Hi!")
    ]
    expected_output = (
        "<s>[INST] <<SYS>>\n"
        "You are an assistant.\n"
        "<</SYS>>\n\n"
        "Hi! [/INST]"
    )
    assert generate_llama_prompt(messages) == expected_output

def test_prompt_for_multi_turn_dialog():
    messages=[
        Message.from_system("You are a helpful assistant."),
        Message.from_user("Hi!"),
        Message.from_assistant("Hello"),
        Message.from_assistant("How are you?"),
        Message.from_user("Great"),
        Message.from_user("Can you help me?"),
        Message.from_assistant("Sure, what should I do?"),
        Message.from_user("Turn on the lights!")
    ]
    expected_output = (
        "<s>[INST] <<SYS>>\n"
        "You are a helpful assistant.\n"
        "<</SYS>>\n\n"
        "Hi! [/INST] "
        "Hello How are you? </s><s>[INST] "
        "Great Can you help me? [/INST] "
        "Sure, what should I do? </s><s>[INST] "
        "Turn on the lights! [/INST]"
    )
    assert generate_llama_prompt(messages) == expected_output

def test_prompt_with_multiple_system_messages():
    messages=[
        Message.from_system("You are a smart assistant."),
        Message.from_user("Hi, how are you?"),
        Message.from_system("Always be polite."),
        Message.from_assistant("Good and you?"),
        Message.from_user("Great"),
        Message.from_system("And Awesome."),
    ]
    expected_output = (
        "<s>[INST] <<SYS>>\nYou are a smart assistant.\n\nAlways be polite.\n\nAnd Awesome.\n<</SYS>>\n\n"
        "Hi, how are you? [/INST] "
        "Good and you? </s><s>[INST] "
        "Great [/INST]"
    )
    assert generate_llama_prompt(messages) == expected_output
  
def test_prompt_fails_with_no_system_message():
    messages=[
        Message.from_user("Hi!")
    ]
    with pytest.raises(ValueError):
        generate_llama_prompt(messages)

def test_prompt_without_user_message():
    messages=[
        Message.from_system("You are an assistant."),
    ]
    expected_output = (
        "<s>[INST] <<SYS>>\n"
        "You are an assistant.\n"
        "<</SYS>>\n\n[/INST]"
    )
    assert generate_llama_prompt(messages) == expected_output


def test_parse_response():
    text = " Good, how are you?  </s> "
    msg = parse_llama_response(text)
    assert msg.role == Role.ASSISTANT
    assert msg.content == "Good, how are you?"

def test_parse_without_end_tag():
    text = "While this misses </s> at the end, it should still"
    msg = parse_llama_response(text)
    assert msg.role == Role.ASSISTANT
    assert msg.content == text
