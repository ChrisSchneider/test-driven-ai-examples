import time
import streamlit as st
from lib.model import Message, Role
from lib.watsonx_llm import WatsonxLLM

# Init session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        Message.from_system("You are a helpful assistant. Be short and precise."),
        Message.from_assistant("Hello, how can I help?"),
    ]
if "llm" not in st.session_state:
    st.session_state.llm = WatsonxLLM()

# Header
st.subheader("Smart Assistant")

# Message list
for msg in st.session_state.messages:
    if msg.role != Role.SYSTEM:
        with st.chat_message(msg.role):
            st.markdown(msg.content)

# Chat input
if prompt := st.chat_input("Your question"):
    # Add user message:
    usr_msg = Message.from_user(prompt)
    with st.chat_message("user"):
        st.markdown(usr_msg.content)
    st.session_state.messages.append(usr_msg)
    
    # Get reply in streaming fashion
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        for res_msg in st.session_state.llm.complete_chat(st.session_state.messages, stream=True):
            message_placeholder.markdown(res_msg.content + "▌")
        message_placeholder.markdown(res_msg.content)
        st.session_state.messages.append(res_msg)
