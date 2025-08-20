#   Phase 1

# Defines how the AI chatbot (agent) is created using LangGraph and how it responds to inputs

# Step 1: Setup API keys for Grok and Tavily

import os
#loading APIs from .env file
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY=os.environ.get('GROQ_API_KEY') # from 'https://console.groq.com/keys'
TAVILY_API_KEY=os.environ.get('TAVILY_API_KEY') # from 'https://app.tavily.com/home' 
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY') # from openai keys section

# Step 2: Setup LLm and tools

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults # Tavily is used for live searches results

groq_llm=ChatGroq(model='llama-3.3-70b-versatile')
openai_llm=ChatOpenAI(model='GPT-4 Omni')
search_tool=TavilySearchResults(max_results=3) #Adds live search capability if allow_search=True

# Step 3: Setup AI agent with search tool functionality
from langgraph.prebuilt import create_react_agent
#This is a React-style reasoning agent that chains reasoning steps using tools and prompts.
#first Reason , then Act agents

from langchain_core.messages.ai import AIMessage
system_prompt='Act as  an AI chatbot who is smart and funny'

def get_response_from_ai_agent(llm_id,query,allow_search,system_prompt,provider):#llm_id= which model from provider 
    if provider=='Groq':
        llm=ChatGroq(model=llm_id)
    elif provider=='OpenAI':
        llm=ChatOpenAI(model=llm_id)

    tools=[TavilySearchResults(max_results=3)] if allow_search else []  

    agent=create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=system_prompt #this gives behavior/attitude to the AI (e.g., "Act as a smart and funny AI")
    )

    #The agent expects the input messages as a dictionary with a 'messages' key. 
    #You wrap the user’s query (which should be a list of strings)
    state={'messages':query}
    response=agent.invoke(state) #triggers the agent to process the conversation using its model and tools, based on the prompt.
    messages=response.get('messages')
    ai_messages=[message.content for message in messages if isinstance(message,AIMessage) ]
    final_response=ai_messages[-1]
    return final_response


# uncomment below to verify if agent is working or not 

# from langgraph.prebuilt import create_react_agent

# from langchain_core.messages.ai import AIMessage
# system_prompt='Act as  an AI chatbot who is smart and funny'
# agent=create_react_agent(
#         model=groq_llm,
#         tools=[search_tool],
#         state_modifier=system_prompt
#     )
# query='Tell me about trends in crypto market!'
# state={'messages':query}
# response=agent.invoke(state)
# messages=response.get('messages')
# ai_messages=[message.content for message in messages if isinstance(message,AIMessage) ]
# print(ai_messages[-1])
