import streamlit as st
from agent_manager import AgentManager
import openai
import asyncio

# --- CONFIG ---
st.set_page_config(page_title="WanderRoo", page_icon="🦘", layout="wide")

# --- API KEY ---
openai.api_key = st.secrets["openai"]["api_key"]

# --- INIT CHAT HISTORY ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "G'day mate! I'm WanderRoo, your local guide to the best stays in Sydney 🏖️. What kind of getaway are ya planning today?"
        }
    ]

# --- INIT AGENT MANAGER ---
if "agent_manager" not in st.session_state:
    st.session_state.agent_manager = AgentManager(api_key=openai.api_key)

# --- CHAT UI ---
st.title("🦘 WanderRoo - Plan Your Dream Sydney Stay")

# --- Display chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask about rentals, availability or reviews...")

if user_input:
    # Add user input to message history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    response_parts = []

    async def get_response():
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            # Stream the response using the updated AgentManager method
            async for part in st.session_state.agent_manager.process_user_query(user_input):
                response_parts.append(part)
                message_placeholder.markdown(''.join(response_parts))

    # Run the response retrieval asynchronously
    asyncio.run(get_response())

    # Combine parts and append full response to message history
    full_response = ''.join(response_parts)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
