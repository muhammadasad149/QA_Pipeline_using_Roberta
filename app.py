import streamlit as st
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.readers import ExtractiveReader
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.writers import DocumentWriter

# Load document
loader = Docx2txtLoader("IGI LIFE WTO Zeenat Takaful Plan.docx")
data = loader.load()
data_content = data[0].page_content

# Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=30,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.create_documents([data_content])

# Prepare Haystack components
model = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
# model = "deepset/roberta-base-squad2"
document_store = InMemoryDocumentStore()

# Embedding pipeline
indexing_pipeline = Pipeline()
indexing_pipeline.add_component(instance=SentenceTransformersDocumentEmbedder(model=model), name="embedder")
indexing_pipeline.add_component(instance=DocumentWriter(document_store=document_store), name="writer")
indexing_pipeline.connect("embedder.documents", "writer.documents")
textss = [Document(content=doc.page_content, meta=doc.metadata) for doc in texts]
indexing_pipeline.run({"documents": textss})

# Initialize retriever and reader
retriever = InMemoryEmbeddingRetriever(document_store=document_store)
reader = ExtractiveReader(model="deepset/roberta-base-squad2")
reader.warm_up()

# Create QA pipeline
extractive_qa_pipeline = Pipeline()
extractive_qa_pipeline.add_component(instance=SentenceTransformersTextEmbedder(model=model), name="embedder")
extractive_qa_pipeline.add_component(instance=retriever, name="retriever")
extractive_qa_pipeline.add_component(instance=reader, name="reader")
extractive_qa_pipeline.connect("embedder.embedding", "retriever.query_embedding")
extractive_qa_pipeline.connect("retriever.documents", "reader.documents")


st.set_page_config(page_title="Document QA Chatbot", page_icon="🤖")
# Streamlit Chatbot UI
st.title("Document QA Chatbot")

# Initialize session state for storing chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat history display in the chat format
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Text input at the bottom of the page
if prompt := st.chat_input("Ask a question:"):
    # Append user query to messages
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run the extractive QA pipeline
    answer = extractive_qa_pipeline.run(
        data={"embedder": {"text": prompt}, "retriever": {"top_k": 5}, "reader": {"query": prompt, "top_k": 1}}
    )

    # Extract content from answer
    extracted_content = []
    for ans in answer['reader']['answers']:
        if ans.data and ans.document:
            extracted_content.append(ans.document.content)

    bot_answer = extracted_content[0] if extracted_content else "No relevant information found."

    # Append the bot's answer to messages
    st.session_state.messages.append({"role": "assistant", "content": bot_answer})

    with st.chat_message("assistant"):
        st.markdown(bot_answer)
