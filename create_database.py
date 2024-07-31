import os
import shutil
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings

def get_embedding_function():
    embeddings = OllamaEmbeddings(model="shaw/dmeta-embedding-zh:latest")
    return embeddings

def get_text_splitter():
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\uff0c",  # Fullwidth comma
            "\u3001",  # Ideographic comma
            "\uff0e",  # Fullwidth full stop
            "\u3002",  # Ideographic full stop
            "",
        ],
        chunk_size=500,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter

def clear_database(path):
    if os.path.exists(path):
        shutil.rmtree(path)

def create_database(data_path, summary_path, data_db_path, summary_db_path, flag = ""):
    if flag == "--reset":
        print("Clearing database...")
        clear_database(data_db_path)
        clear_database(summary_db_path)
    # original data
    documents = load_documents(data_path)
    chunks = split_documents(documents)
    add_to_chroma(chunks, data_db_path)
    # summary data
    text = text_from_file(summary_path)
    text_chunks = split_text(text)
    text_to_chroma(text_chunks, summary_db_path)
    # print(text_chunks[0])
    
def text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text
    
def load_documents(path):
    document_loader = PyPDFDirectoryLoader(path)
    documents = document_loader.load()
    return documents

def split_text(text):
    text_splitter = get_text_splitter()
    text_chunks = text_splitter.split_text(text)
    return text_chunks

def split_documents(documents):
    text_splitter = get_text_splitter()
    chunks = text_splitter.split_documents(documents)
    return chunks

def text_to_chroma(text_chunks, db_path):
    db = Chroma(
        persist_directory=db_path, embedding_function=get_embedding_function()
    )
    if len(text_chunks):
        print("adding summary to db...")
        db.add_texts(text_chunks)
        
def add_to_chroma(chunks, db_path):
    # Load the existing database.
    db = Chroma(
        persist_directory=db_path, embedding_function=get_embedding_function()
    )
    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("No new documents to add")


def calculate_chunk_ids(chunks):
    # Page Source : Page Number : Chunk Index
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks