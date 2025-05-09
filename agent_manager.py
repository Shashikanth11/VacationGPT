import re
import openai
from openai import OpenAI 
from agents import Agent, FileSearchTool, enable_verbose_stdout_logging, Runner, ItemHelpers
import streamlit as st
import json
import os
import asyncio
import concurrent.futures
#from openai.types.responses import RunItemStreamEvent, MessageOutputItem



#enable_verbose_stdout_logging()

class AgentManager:
    def __init__(self, api_key=None, user=None):
        """Initialize the agent manager"""
        self.api_key = api_key
        # if api_key:
        #     os.environ["OPENAI_API_KEY"] = api_key
        self.client = None
        self.triage_agent = None
        self.agents = {}
        self.user = user
        self.conversation_history = [{"role": "system", "content": "You are a helpful assistant."}] #codoc


        # Initialize vector stores
        self.listings_vector_store = None
        self.reviews_vector_store = None

    def _ensure_client(self):
        """Ensure the OpenAI client is initialized with the API key"""
        if not self.api_key:
            st.error("OpenAI API key must be provided to use this functionality")
            return None

        if not self.client:
            try:
                os.environ["OPENAI_API_KEY"] = self.api_key
                self.client = OpenAI(api_key=self.api_key)
                print(f"OpenAI client initialized successfully with API key starting with: {self.api_key[:5]}...")
            except Exception as e:
                st.error(f"Failed to initialize OpenAI client: {str(e)}")
                return None

        return self.client

    def _load_vector_stores(self):
        """Load the vector stores for listings and reviews from OpenAI or session"""
        try:
            # Assuming the vector stores are uploaded as individual files to OpenAI
            self.listings_vector_store = openai.File.retrieve(self.listings_vector_store_id)
            self.reviews_vector_store = openai.File.retrieve(self.reviews_vector_store_id)

            if not self.listings_vector_store:
                st.error("Listings vector store not found in OpenAI.")
                return False

            if not self.reviews_vector_store:
                st.error("Reviews vector store not found in OpenAI.")
                return False

            print("Vector stores loaded successfully from OpenAI")
            return True
        except Exception as e:
            st.error(f"Error loading vector stores from OpenAI: {str(e)}")
            return False

    def initialize_agents(self):
        """Initialize all agents in the system"""
        # if not self._ensure_client():
        #     return None

        # if not self._load_vector_stores():
        #     return None  # Return early if vector stores are not loaded

        # if self.triage_agent:
        #     return self.triage_agent

        try:
            # self.agents["listings_agent"] = self._create_listings_agent()
            # self.agents["reviews_agent"] = self._create_reviews_agent()

            # self.triage_agent = self._create_triage_agent([
            #     self.agents["listings_agent"],
            #     self.agents["reviews_agent"]
            # ])
            self.triage_agent = self._create_triage_agent()

            print("All agents initialized successfully")
            return self.triage_agent
        except Exception as e:
            error_msg = f"Error initializing agents: {str(e)}"
            print(error_msg)
            st.error(error_msg)
            return None

    def _create_listings_agent(self):
        """Create agent for Airbnb listings retrieval"""
        return Agent(
            name="Airbnb listings filtering",
            model="gpt-4o",
            instructions="""You are the Listings Agent responsible for retrieving relevant Airbnb listings based on the user's preferences.
You will receive a user query and use the provided vector store to find the best matching listings.
Your tasks are:
1. Parse the user's query for location, number of people, and other preferences (e.g., pool, pet-friendly).
2. Search through the vector store to find listings that match the user's criteria.
3. Return a list of filtered listings with brief details (e.g., name, location, price, key amenities) and a unique listing_id for each listing.
4. If no relevant listings are found, inform the user and suggest refining the search.

For example:
User Query: "Find me a 2-bedroom apartment in Sydney with a pool for 2 adults."
- You should search the vector store for apartments in Sydney, ensuring it meets the number of bedrooms, has a pool, and accommodates 2 adults.

If a property matches the user's preferences, return the property details like:
- Property Name: "Sunny Apartment"
- Location: "Sydney, Australia"
- Price: "$200 per night"
- Amenities: "Pool, Wi-Fi, Air Conditioning"
- Listing ID: "12345"  # Unique identifier

If no listings match, respond with a message like: "Sorry, no listings matched your preferences. Please refine your search criteria."
        """,
            tools=[
                FileSearchTool(vector_store_ids=[self.listings_vector_store.id])
            ]
        )

    def _create_reviews_agent(self):
        """Create agent for Airbnb reviews retrieval"""
        return Agent(
            name="Airbnb reviews summarisation",
            model="gpt-4o",
            instructions="""You are the Reviews Agent responsible for summarizing the reviews of a specific Airbnb property.
You will receive a user query that may include a property name or ID, and you will search the reviews vector store to provide a summary.
Your tasks are:
1. Parse the user's query to identify the property the user is asking about.
2. Use the provided `listing_id` from the Listings Agent to search the reviews vector store for the relevant reviews.
3. Provide a summary of the reviews, highlighting key points such as overall satisfaction, strengths (e.g., "great location", "comfortable bed"), and weaknesses (e.g., "noisy neighborhood", "small bathroom").
4. If there are no reviews available or the property cannot be found, inform the user.

For example:
User Query: "What do people say about the 'Sunny Apartment' in Sydney?"
- You should search the reviews vector store for reviews related to the provided `listing_id` of "Sunny Apartment".

If reviews are found, summarize them like:
- "The property is highly rated for its great location and comfortable furnishings, but some guests noted that it can get noisy at night due to traffic."

If no reviews are available, respond with something like: "Sorry, no reviews are available for this property."
        """,
            tools=[
                FileSearchTool(vector_store_ids=[self.reviews_vector_store.id])
            ]
        )

    def _create_triage_agent(self):#(self, specialized_agents):
        """Create the main triage agent"""
        return Agent(
            name="User query management",
            model="gpt-4o",
            instructions="""You are the triage agent responsible for determining the user's needs. You respond in Aussie slang.
Your job is to:
1. Identify if the user is asking for information about:
   - Airbnb listings (e.g., searching for properties, finding specific amenities)
   - Reviews (e.g., summarizing reviews for a listing)
2. Route the user query to the appropriate specialized agent:
   - If the query is about listings (e.g., "Show me available properties in Sydney for 2 adults"), forward it to the Listings Agent.
   - If the query is about reviews (e.g., "What do people say about this property?"), forward it to the Reviews Agent.
3. If the query is unclear or needs further clarification, ask the user for more information before routing the request.

For example:
User Query: "Find me a 2-bedroom apartment in Sydney with a pool."
- This should be routed to the Listings Agent.

User Query: "What are the reviews for this property?"
- This should be routed to the Reviews Agent.
            """,
            # handoffs=specialized_agents #,
            #tools=[function_tool(self.get_user_profile)]
        )

    async def process_user_query(self, user_query):
        try:
            if not self._ensure_client():
                yield "Client not ready."
                return

            if not self.triage_agent :
                if not self.initialize_agents():
                    yield "Initialization failed."
                    return

            self.conversation_history.append({"role": "user", "content": user_query})


            response =  Runner.run_streamed(
                starting_agent=self.triage_agent,
                input=self.conversation_history
            )

            full_response_parts = []

            async for event in response.stream_events():
                 if event.type == "run_item_stream_event":
                     item = getattr(event, "item", None)
                     if item and hasattr(item, "raw_item"):
                         content_blocks = getattr(item.raw_item, "content", [])
                         for block in content_blocks:
                             if hasattr(block, "text"):
                                 yield block.text
                                 full_response_parts.append(block.text)
    
            full_response = "".join(full_response_parts)
                # Add assistant reply to conversation history
            self.conversation_history.append({"role": "assistant", "content": full_response})

        except Exception as e:
            print(f"Error while processing user query: {e}")
            yield f"Error: {e}"

