import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from agent.tools import check_properties_tool
import toml
import os

# Load API key from 'chainlit/secrets.toml'
secrets_path = os.path.join(os.path.dirname(__file__), "chainlit", "secrets.toml")
secrets = toml.load(secrets_path)
API_KEY = secrets.get("api_key")

if not API_KEY or not API_KEY.startswith("sk-"):
    raise ValueError("❌ Invalid or missing OpenAI API key in chainlit/secrets.toml")

@cl.on_chat_start
async def start():
    # Initialize LLM
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo",
        openai_api_key=API_KEY
    )

    # Create LangChain agent
    agent = initialize_agent(
        tools=[check_properties_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    # Store agent in session
    cl.user_session.set("agent", agent)

    # Show assistant name centered using Markdown
    await cl.Message(content="### 🏖️ **VacationGPT**", author=None).send()
    await cl.Message(
        content="Hi! I'm your travel assistant. Ask me anything about Airbnb properties in Sydney!",
        author="VacationGPT"
    ).send()

@cl.on_message
async def handle_message(msg: cl.Message):
    agent = cl.user_session.get("agent")

    if not agent:
        await cl.Message(
            content="❌ Agent not initialized. Please refresh and try again.",
            author="VacationGPT"
        ).send()
        return

    response = agent.run(msg.content)
    await cl.Message(content=response, author="VacationGPT").send()
