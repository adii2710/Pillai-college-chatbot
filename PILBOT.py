from huggingface_hub import InferenceClient
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import faiss
import pickle

index=faiss.read_index("docsind.index")
with open("faiss_store.pkl", "rb") as f:
    store=pickle.load(f)

store.index=index

# Your predefined template
template = """
You are a chatbot assistant by Pillai College of Engineering that provides information about student services and the college.
If you don't know the answer, just say "sorry..!, I'm not sure about the answer. Please visit the website for further assistance." 
Don't try to make up an answer.
CHAT HISTORY: {chat_history}

HUMAN: {question}
=========
{summaries}
=========
CHATBOT:
"""

# Function to generate the prompt using the template, chat history, and retrieved documents
def generate_prompt(question, retrieved_docs, chat_history):

    # Combine previous conversation (chat history)
    if len(chat_history) != 0:
        history_text = "\n".join([f"HUMAN: {item['question']}\nCHATBOT: {item['answer']}" for item in chat_history])
    else: 
        history_text=None
    
    # Combine the retrieved documents from vector store (if any)
    doc_summaries = "\n".join([f"CONTENT: {doc.page_content}" for doc in retrieved_docs])

    
    # Fill in the template with chat history, question, and document summaries
    prompt = template.format(question=question, summaries=doc_summaries, chat_history=history_text)
    
    return prompt


# Function to query LLaMA with the generated prompt

def query_llama(prompt):
    hf_token='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'
    client = InferenceClient(
        "mistralai/Mistral-Nemo-Instruct-2407",
        token=hf_token,
)
    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        stream=False  # You can use streaming if you prefer
    )
    return response['choices'][0]['message']['content']

# Main function to search vector store, generate prompt using the template, and query LLaMA
def query_with_template_and_sources(question, vectorstore):
    chat_history=[]
    # Retrieve relevant documents from vector store
    docs = vectorstore.similarity_search(question)
    # with open("testing.txt", 'a') as fp:
    #     fp.write(str(docs))
    
    # Generate the prompt using the template, including chat history and document summaries
    prompt = generate_prompt(question, docs, chat_history)
    
    # Query LLaMA model with the generated prompt
    answer = query_llama(prompt)
    
    # Add the current question and answer to chat history
    chat_history.append({"question": question, "answer": answer})
    with open("testing.txt", 'a') as fp:
        fp.write(str(chat_history)+"\n")
    
    return answer

st.title("PILBOT")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages=[]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt=st.chat_input("what's up?")

if prompt:
    with st.chat_message('user'):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role":"user", "content": prompt})

    response= f"Echo: {query_with_template_and_sources(prompt, store)}"
    # display assitant response
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # add assitant response to chat history
    st.session_state.messages.append({"role":"assistant", "content":response})
