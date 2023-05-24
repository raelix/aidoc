#!/bin/python3
# Simple script to run a AI Doc with a UI using streamlit
# run:
# streamlit run aidoc.py
#
import streamlit as st
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent, initialize_agent, AgentType
from langchain.chains import LLMChain,RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import GoogleSearchAPIWrapper
import os

index_name = os.environ['PINECONE_INDEX_NAME']

def initDep():
    OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
    PINECONE_API_KEY = os.environ['PINECONE_API_KEY']
    PINECONE_API_ENV = os.environ['PINECONE_API_ENV']
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_API_ENV
    )
    return embeddings, llm


embeddings, llm = initDep()

instance = Pinecone.from_existing_index(
    index_name=index_name, embedding=embeddings)


# Chain for question-answering against an index
os.environ["GOOGLE_CSE_ID"] = os.environ["AI_GOOGLE_ID"]
os.environ["GOOGLE_API_KEY"] = os.environ["AI_GOOGLE_API_KEY"]
search = GoogleSearchAPIWrapper()

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=instance.as_retriever()
)

# Tool that takes in function or coroutine directly.

tools = [
    Tool(
        name="DocumentationDB",
        func=qa.run,
        description="Useful for when you need to answer questions about the aspects asked. Input may be a partial or fully formed question.",
    ),
    # Tool(
    #     name="Google Search",
    #     description="Search Google if you don't find anything on DocumentationDB.",
    #     func=search.run
    # ),
]

prefix = """Have a conversation with a human, answering the following questions based on the context and memory available, output must be Markdown format. 
                You have access to one tool:"""

suffix = """Begin!"
{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)

st.title("AI Doc") 

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history"
    )

llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
)

agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)

agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=st.session_state.memory
)

query = st.text_input(
    "**Ask me a question**",
    placeholder="...Anything about your docs",
)

if query:
    with st.spinner(
        "Generating answer to your query : `{}` ".format(query)
    ):
        res = agent_chain.run(query)
        st.markdown(res)  # icon="🤖"

with st.expander("History"):
    st.session_state.memory
