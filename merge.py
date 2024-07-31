from langchain_community.llms.ollama import Ollama
from langchain.prompts import ChatPromptTemplate
model = Ollama(model="llama3.1")

PROMPT_TEMPLATE = """
merge the following two parts of text into one in markdown format:
First part:
{first}
Second part:
{second}
"""

def merge(response_1, response_2):
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(first = response_1, second = response_2)
    response_text = model.invoke(prompt)
    return response_text