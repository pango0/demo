import create_database
from merge import merge
import argparse
from summarize import summarize
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

def parse_arguments():
    parser = argparse.ArgumentParser(description="A script with -q and -c flags")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-q", "--query", action="store_true", help="Query question")
    group.add_argument("-c", "--create_database", action="store_true", help="Create vector database")
    return parser.parse_args()


PROMPT_TEMPLATE = """
僅根據以下提供的內容回答問題：

{context}

---

僅根據以上提供的內容回答問題，如果問題與內容無關就回答"此問題我無法回答"就好，不要回覆額外內容: {question}
"""

def query_rag(query_text: str, db_path: str):
        embedding_function = create_database.get_embedding_function()
        db = Chroma(persist_directory=db_path, embedding_function=embedding_function)
        results = db.similarity_search_with_score(query_text, k=10)

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        model = Ollama(model="jcai/llama3-taide-lx-8b-chat-alpha1:Q4_K_M")
        response_text = model.invoke(prompt)
        formatted_response = f"Response: {response_text}"
        print("\n"+formatted_response)
        return response_text
    
def main():
    args = parse_arguments()
    if args.create_database:
        summarize("data")
        create_database.create_database("data", "summaries/summary.txt", "original", "summary")
    elif args.query:
        query = input("請問您想問專家甚麼?")
        # print("Summary response:")
        summarized = query_rag(query, "./summary")
        # print("Original response:")
        original = query_rag(query, "./original")
        print("Final response:")
        final = merge(summarized, original)
        print(final)
    
if __name__ == "__main__":
    main()
    