#phase 3:

# A user interface to allow input of model parameters and queries, and display AI responses

# Step 1: Setup UI with streamlit (model_provider,model,system_prompt,web_search,query)

import streamlit as st

st.set_page_config(page_title='Moeen LangGraph Agent UI',layout='centered')
st.title('Moeen AI chatbot agent')
st.write('Create and interact woth AI agents')

system_prompt=st.text_area('Define your AI agent:',height=70,placeholder='Type your system prompt here...')

MODEL_NAMES_GROQ=['llama-3.3-70b-versatile','mixtral-8x7b-32768']
MODEL_NAMES_OPENAI=['gpt-4o-mini','GPT-4 Omni']

provider= st.radio('Select provider:',{'Groq','OpenAI'})

if provider=='Groq':
    selected_model=st.selectbox('Select Groq model:',MODEL_NAMES_GROQ)
elif provider=='OpenAI':
    selected_model=st.selectbox('Select OpenAI model:',MODEL_NAMES_OPENAI)  

allow_web_search=st.checkbox('Allow web search')      

user_query=st.text_area('Enter your query:',height=69,placeholder='Ask anything!')

API_URL= 'http://127.0.0.1:8989/chat'

if st.button('Ask Agent'):
    if user_query.strip(): # if querry is not empty or filled with spaces

        import requests

        # Step 2: Connect with backend via URL 

        #Payload creation: The inputs are collected into a JSON dictionary 
        payload={
            'model_name': selected_model,
            'model_provider': provider,
            'system_prompt': system_prompt,
            'messages': [user_query],
            'allow_search': allow_web_search
        }

        # Sends the data to FastAPI /chat endpoint and receives a JSON response
        response=requests.post(API_URL,json=payload)

        # # Get response from backend and show here
        # response='This is just a shtendu response'   #no need of this now

        if response.status_code==200:
            response_data=response.json()
            if 'error' in response_data:
                st.error(response_data['error'])
            else:
                st.subheader('Agent response')
                st.markdown(f'**Final Response:** {response_data}')
    else:
        st.subheader('Agent response:')
        st.markdown(f'**Final Response:** Kindly enter your querry')    
        
