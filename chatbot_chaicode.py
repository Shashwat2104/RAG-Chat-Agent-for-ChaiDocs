# chatbot_chaicode.py

from langchain.document_loaders import WebsiteLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import pinecone
import os

# Set API Keys
os.environ["OPENAI_API_KEY"] = "your-openai-key"
pinecone.init(api_key="your-pinecone-key", environment="your-pinecone-environment")

# Step 1: Load Documentation
print("🔍 Loading ChaiCode Documentation...")
loader = WebsiteLoader("https://chaicode.dev/docs", max_depth=3)
documents = loader.load()

# Step 2: Chunk the Docs
print("📄 Chunking Text...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(documents)

# Step 3: Embed & Store in Pinecone
print("📦 Creating Embeddings & Storing in Pinecone...")
embeddings = OpenAIEmbeddings()
index_name = "chaicode-docs"

# Create index if not exists
if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, dimension=1536, metric="cosine")

# Store vectors
vectorstore = Pinecone.from_documents(chunks, embeddings, index_name=index_name)

# Step 4: Set up Retrieval-QA Chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0),
    retriever=retriever,
    return_source_documents=True
)

# Step 5: User Query (you can turn this into a loop/UI later)
while True:
    user_input = input("\nAsk about ChaiCode Docs (or 'exit'): ")
    if user_input.lower() == 'exit':
        break

    result = qa_chain({"query": user_input})
    print("\n💬 Answer:\n", result["result"])
    print("\n🔗 Sources:")
    for doc in result["source_documents"]:
        print("-", doc.metadata.get("source", "No URL Found"))
