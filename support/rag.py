import os
from pypdf import PdfReader
from decouple import config

from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.documents import Document

#  initialize chromadb client
# client = chromadb.PersistentClient(path="./chroma_db")


# default embedding model that converts text into numbers
# embedding_fn = DefaultEmbeddingFunction()  


# embedding_function = SentenceTransformerEmbeddingFunction(
#     model_name="all-MiniLM-L6-v2"
# )

#  get or create collection - just like table in regular database
# collection = client.get_or_create_collection(
#     name="coolbreeze_docs",
#     embedding_function= embedding_function
# )

# embeddings = MistralAIEmbeddings(
#     model="mistral-embed",
#     api_key=config("MISTRAL_API_KEY"),
# )



# vector_store = Chroma(
#     collection_name="coolbreeze_docs",
#     embedding_function=embeddings,
#     persist_directory="./chroma_db",
# )

# def chunk_text(text, chunk_size=500):
#     words = text.split()
#     chunks = []
#     current_chunk = []
#     current_size = 0

#     for word in words:
#         current_chunk.append(word)
#         current_size += len(word) + 1

#         if current_size >= chunk_size:
#             chunks.append(" ".join(current_chunk))
#             current_chunk = []
#             current_size = 0

#     if current_chunk:
#         chunks.append(" ".join(current_chunk))

#     return chunks


# def load_documents():
#     collection = get_collection()

#     if collection.count() > 0:
#         print(f"Knowledge base already exists ({collection.count()} chunks).")
#         return

#     docs_path = "support/documents"

#     if not os.path.exists(docs_path):
#         raise FileNotFoundError(f"{docs_path} not found")

#     documents = []
#     ids = []

#     for filename in os.listdir(docs_path):
#         if filename.endswith(".pdf"):

#             file_path = os.path.join(docs_path, filename)
#             reader = PdfReader(file_path)

#             full_text = ""
#             for page in reader.pages:
#                 full_text += page.extract_text()

#             chunks = chunk_text(full_text, chunk_size=500)

#             for i, chunk in enumerate(chunks):
#                 documents.append(chunk)
#                 ids.append(f"{filename}_{i}")

#     if documents:
#         collection.add(documents=documents, ids=ids)

#     print(f"Loaded {len(documents)} chunks into ChromaDB.")



# def search_knowledge_base(query):
#     collection = get_collection()

#     results = collection.query(query_texts=[query], n_results=3)

#     if not results["documents"][0]:
#         return "No relevant information found in company documents."

    # matched_chunks = results["documents"][0]
#     return "\n\n".join(matched_chunks)


# -----------------------------
# Lazy Initialization
# -----------------------------
embeddings = None
vector_store = None


def get_vector_store():
    global embeddings, vector_store

    if vector_store is None:
        embeddings = MistralAIEmbeddings(
            model="mistral-embed",
            api_key=config("MISTRAL_API_KEY"),
        )

        vector_store = Chroma(
            collection_name="coolbreeze_docs",
            embedding_function=embeddings,
            persist_directory="./chroma_db",
        )

    return vector_store


# -----------------------------
# Text Chunking
# -----------------------------
def chunk_text(text, chunk_size=500):
    words = text.split()

    chunks = []
    current_chunk = []
    current_size = 0

    for word in words:
        current_chunk.append(word)
        current_size += len(word) + 1

        if current_size >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_size = 0

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# -----------------------------
# Load PDFs into Chroma
# -----------------------------
def load_documents():
    vector_store = get_vector_store()

    docs_path = "support/documents"

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"{docs_path} not found")

    documents = []

    for filename in os.listdir(docs_path):

        if not filename.endswith(".pdf"):
            continue

        file_path = os.path.join(docs_path, filename)

        reader = PdfReader(file_path)

        full_text = ""

        for page in reader.pages:
            text = page.extract_text()

            if text:
                full_text += text

        chunks = chunk_text(full_text)

        for i, chunk in enumerate(chunks):

            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": filename,
                        "chunk": i,
                    },
                )
            )

    if documents:
        vector_store.add_documents(documents)

    print(f"Loaded {len(documents)} chunks into ChromaDB")


# -----------------------------
# Search
# -----------------------------
def search_knowledge_base(query):
    vector_store = get_vector_store()

    results = vector_store.similarity_search(query, k=3)

    if not results:
        return "No relevant information found in company documents."

    return "\n\n".join(doc.page_content for doc in results)