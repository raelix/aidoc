#!/bin/python3
# A simple script to vectorize PDF files to Pinecone

import pinecone
import os
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
  return embeddings,llm

def load(filename):
  loader = UnstructuredPDFLoader(filename)
  data = loader.load()
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
  texts = text_splitter.split_documents(data)
  Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)

embeddings, llm = initDep()

# vectorize the PDFs
# provide the pdf to vectorize
# load("./my.pdf")

