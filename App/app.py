import streamlit as st
from ollama import Client
from prompt import Prompt
from typing import Generator
from search import Search
from nifty_50 import nifty_mappings
import os

p = Prompt()
s = Search()
llm = Client(host=f'http://{os.getenv("OLLAMA_HOST", "localhost")}:11434')
MODEL = os.getenv('LLM_MODEL', 'phi3')
print(f"Using model: {MODEL} on Server at http://{os.getenv('OLLAMA_HOST', 'localhost')}:11434")


def ollama_stream_generator(query: str, context: str, company: str, fy: int or str) -> Generator:
    stream = llm.chat(model=MODEL,
                  messages=p.get_prompt(query,
                                        context,
                                        company,
                                        fy),
                  stream=True)
    for chunk in stream:
        yield chunk['message']['content']


st.title("Finchat: NiftyInsights Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

if 'fy' not in st.session_state:
    st.session_state.fy = 'ALL'

if 'company' not in st.session_state:
    st.session_state.company = 'ALL'

company_col, fy_col = st.columns(2)
with company_col:
    company_dropdown = st.selectbox("Choose your company", list(nifty_mappings.keys()))
    st.session_state.company = company_dropdown
with fy_col:
    fy_dropdown = st.selectbox("Choose your FY", [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023])
    st.session_state.fy = fy_dropdown

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("How can I help you?"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    context = s.get_similar_docs(query_text=user_query,
                                 symbol=st.session_state.company if not st.session_state.company == 'ALL' else None,
                                 fy=st.session_state.fy if not st.session_state.fy == 'ALL' else None)

    print("***** . context: ", context.strip())

    with st.chat_message("assistant"):
        response = st.write_stream(ollama_stream_generator(user_query,
                                                           context,
                                                           st.session_state.company,
                                                           st.session_state.fy))

    st.session_state.messages.append({"role": "assistant", "content": response})
