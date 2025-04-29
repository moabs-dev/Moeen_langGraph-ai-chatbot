# Phase 2

#Handles requests (e.g., from Swagger UI or frontend) and routes them to the AI agent

# step 1: Setup ydantic model(schema validation)

from pydantic import BaseModel
from typing import List

class RequestState(BaseModel): #Validates input structure:model name,provider,prompt,user message and search flag.
    model_name: str
    model_provider: str
    system_prompt: str
    messages: List[str]
    allow_search: bool
    #if these are not true when providing through frontend, there 'll be validation error


# step 2: Setup AI agent from frontend request

from fastapi import FastAPI
from ai_agent import get_response_from_ai_agent

# Ensures only listed model names can be used
ALLOWED_MODEL_NAMES=["llama3-70b-8192", "mixtral-8x7b-32768", "llama-3.3-70b-versatile", "gpt-40-mini","GPT-4 Omni"]
app=FastAPI(title='LangGraph AI agent')


# POST /chat endpoint: Accepts a request body and sends it to the get_response_from_ai_agent(...) function
@app.post('/chat')
def chat_endpoint(request: RequestState):
    '''
    AI Endpoint to interact with the Chatbot using LangGraph and Search tools.
    It dynamically selects the model specified in the request.
    '''

    if request.model_name not in ALLOWED_MODEL_NAMES:
        return {'error':'Invalid model name. KIndly select a valid AI model'}
    
    llm_id = request.model_name 
    query= request.messages
    allow_search= request.allow_search 
    system_prompt = request.system_prompt 
    provider = request.model_provider

    # Create AI agent and get response from it
    response = get_response_from_ai_agent(llm_id,query,allow_search,system_prompt,provider)
    return response

# step 3: Run app and explore swagger UI docs

if __name__=='__main__':
    import uvicorn

    # Starts a local FastAPI server you can interact with at http://127.0.0.1:8989/docs
    uvicorn.run(app,host='127.0.0.1',port=8989)  # add '/docs' on url to see swagger UI 