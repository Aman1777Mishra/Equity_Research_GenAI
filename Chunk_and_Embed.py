from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings 
from langchain_community.vectorstores import FAISS
import web_scraper
import os
import API_keys

os.environ["OPENAI_API_KEY"]=API_keys.open_ai_key

#url = "https://247wallst.com/investing/2026/04/03/price-prediction-nvidia-stock-will-be-worth-this-much-in-2027/"

def chunk_and_embed(document):
    #print(document)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 400,
        chunk_overlap = 10
    )

    chunks = splitter.split_text(document[0].page_content)

    #for i, chunk in enumerate(chunks):
    #    print("\n\n")
    #    print("chunk {i} : \n")
    #    print(chunk)
    
    chunk_doc = [
        Document(
        page_content=chunk,
        metadata = {"source" : "news_article", "chunk_id" : i}
        )
        for i, chunk in enumerate(chunks)
    ]

    embedding = OpenAIEmbeddings()

    vector_db_index = FAISS.from_documents(
        documents=chunk_doc,
        embedding=embedding,
    )

    return vector_db_index
    #vector_db_index.save_local("vector_db_index")
 



        




