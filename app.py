# import 
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain_openai import OpenAIEmbeddings , ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from flask import Flask, request, jsonify, render_template, session
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import requests
from bs4 import BeautifulSoup
import html2text
from langchain.schema import Document 
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

def webscraping(url):
    response = requests.get(url)
    if response.status_code == 500:
        print("Server error")
        return
    
    soup = BeautifulSoup(response.content, 'html.parser')
  
    for script in soup(["script", "style"]):
        script.extract()

    html = str(soup)
    html2text_instance = html2text.HTML2Text()
    text = html2text_instance.handle(html)

    try:
        page_title = soup.title.string.strip()
    except:
        page_title = url.path[1:].replace("/", "-")
    meta_description = soup.find("meta", attrs={"name": "description"})
    meta_keywords = soup.find("meta", attrs={"name": "keywords"})
    if meta_description:
        description = meta_description.get("content")
    else:
        description = page_title
    if meta_keywords:
        meta_keywords = meta_description.get("content")
    else:
        meta_keywords = ""

    metadata = {'title': page_title,
                'url': url,
                'description': description,
                'keywords': meta_keywords}
    return text, metadata

def get_vectorstore(url):
    
    text, metadata = webscraping(url)
    doc_chunks = []
    text_splitter = MarkdownTextSplitter()
    chunks = text_splitter.split_text(text)
    for i, chunk in enumerate(chunks):
        doc = Document(page_content=chunk, metadata=metadata)
        doc_chunks.append(doc)

    
    vector_store = Chroma(
        collection_name="website_data",
        embedding_function=embeddings,
        persist_directory="data/chroma")

    vector_store.add_documents(doc_chunks)
    # vector_store.persist()
    return vector_store

def get_context_chain(vector_store):
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Generate a search query to retrieve relevant information based on the conversation.")
    ])


    retriever_chain = create_history_aware_retriever(llm, retriever , prompt)
    return retriever_chain


def get_convo_chain(retriever_chain):
    llm = ChatOpenAI(api_key="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")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    stuff_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever_chain, stuff_chain)

    return retrieval_chain


def get_response(vector_store, user_input, chat_history):
    retriever_chain = get_context_chain(vector_store)
    convo_chain = get_convo_chain(retriever_chain)
    # Get response from conversation chain
    response = convo_chain.invoke({
        "chat_history": chat_history,
        "input": user_input
    })

    # Handle cases where 'answer' may not exist
    response_text = response.get("answer", "No response generated.")
    
    # Append to chat history
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": response_text})

    return response_text, chat_history


if __name__ == "__main__":
    chat_history = ["Hello! I am a chatbot assistant. Ask me anything!"]

    # Step 1: Take URL input
    url = input("Enter the website URL to scrape: ").strip()
    print(f"Scraping and processing the URL: {url}...")

    # Create vector store from the given URL
    try:
        vector_store = get_vectorstore(url)
        print("Vector store successfully created.")
    except Exception as e:
        print(f"Error while processing the URL: {e}")
        exit()

    # Step 2: Chat with the model
    print("\nYou can now ask questions based on the scraped content. Type 'exit' to quit.")
    while True:
        user_input = input("\nYour Question: ").strip()
        if user_input.lower() == "exit":
            print("Exiting the chatbot. Goodbye!")
            break

        try:
            response, chat_history = get_response(vector_store, user_input, chat_history)
            print(f"Answer: {response}")
        except Exception as e:
            print(f"Error while generating a response: {e}")
