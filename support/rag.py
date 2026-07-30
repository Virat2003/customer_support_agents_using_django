import chromadb
# from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
import os
from pypdf import PdfReader
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


#  initialize chromadb client
client = chromadb.PersistentClient(path="./chroma_db")


# default embedding model that converts text into numbers
# embedding_fn = DefaultEmbeddingFunction()  


embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

#  get or create collection - just like table in regular database
collection = client.get_or_create_collection(
    name="coolbreeze_docs",
    embedding_function= embedding_function
)


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


def load_documents():
    docs_path = "support/documents"

    documents = []
    ids = []

    for filename in os.listdir(docs_path):
        if filename.endswith(".pdf"):
            #  filepath : support/documents/refund_policy.pdf
            file_path = os.path.join(docs_path, filename)
            reader = PdfReader(file_path)

            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text()

            chunks = chunk_text(full_text, chunk_size=500)

            for i, chunk in enumerate(chunks):
                documents.append(chunk)
                ids.append(f"{filename}_{i}")

    if documents:
        collection.add(documents=documents, ids=ids)

    print(f"loaded {len(documents)} chunks into chromadb.")



def search_knowledge_base(query):
    results = collection.query(query_texts=[query], n_results=3)

    if not results["documents"][0]:
        return "No relevant information found in company documents."

    matched_chunks = results["documents"][0]
    return "\n\n".join(matched_chunks)






