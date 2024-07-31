from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
import time
import multiprocessing as mp
import os


llm = Ollama(model="jcai/llama3-taide-lx-8b-chat-alpha1:Q4_K_M")

def doc_summary(docs):
    print(f'You have {len(docs)} document(s)')
    words = [len(doc.page_content.split(' ')) for doc in docs]
    print(words)
    num_words = sum([len(doc.page_content.split(' ')) for doc in docs])
    print(f'You have roughly {num_words} words in your docs')
    print()
    
def preprocess(DATA_PATH):
    loader = PyPDFDirectoryLoader(DATA_PATH)
    docs = loader.load()
    text_splitter = get_text_splitter()
    text = extract_text(docs)
    chunks = text_splitter.split_text(text)
    return chunks    

def parallel(chunks):
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(process_chunk, chunks)
    summary = ''.join(results)
    return summary

def write_to_file(summary):
    os.makedirs("summaries", exist_ok=True)
    summary_file_path = os.path.join("summaries", "summary.txt")
    with open(summary_file_path, "w", encoding="utf-8") as f:
        f.write(summary)

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

def extract_text(docs):
    text = ''
    for doc in docs:
        text = text + doc.page_content
    return text

def process_chunk(chunk):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "幫我用繁體中文整合使用者文字，提供一份完整的markdown重點整理，並去除冗言贅句"),
            ("human", "{text}"),
        ]
    )
    chain = prompt | llm
    ret = chain.invoke({"text": chunk})
    return ret





def summarize(data_path):
    chunks = preprocess(data_path)
    st_time = time.time()
    print("Summarizing...")
    summary = parallel(chunks)
    print("Done")
    print(f"Summarize time: {time.time() - st_time} seconds")
    write_to_file(summary)

    
