import pytest
from unittest.mock import patch
from playwright.sync_api import Page, expect

@pytest.fixture(autouse=True)
def setup(page: Page):
    """Go to app before each test"""
    page.goto("http://localhost:4000/")
    yield

def test_has_title(page: Page):
    """Test if the page has the expected title"""
    h3 = page.locator('h3')
    assert "German Constitution QnA" in h3.text_content()

def test_initial_assistant_message(page: Page):
    """Test if initial message from assistant is shown"""
    expect(page.get_by_label("Chat message from assistant")).to_contain_text("Hello")

def test_new_user_message(page: Page):
    """Test submitting & showing a new user message"""
    chat_input = page.get_by_test_id("stChatInput")
    chat_input.click()
    chat_input.fill("Hello")
    page.keyboard.press("Enter")
    expect(page.get_by_label("Chat message from user")).to_contain_text("Hello")
