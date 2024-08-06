# Build a Retrieval Augmented Generation (RAG) App

#  setup openai environment  
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()
# Retrieve environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# setup langchain environment  
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def setup_Environment():
    
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    return  llm 
    

def load_data():
    # Load, chunk and index the contents of the blog.
    # load  data from  website webscraping 
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()
    return docs

#  format data
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def retrieve(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    
    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    
    rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | setup_Environment()
    | StrOutputParser()
    )
    result = rag_chain.invoke("What is Task Decomposition?")
    print(result)
    
def main():
    setup_Environment()
    docs = load_data()
    retrieve(docs)
    
    print("Done!")
main()
