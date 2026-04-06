from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from API_keys import open_ai_key

def debug_docs(docs):
    print("\n====== RETRIEVED DOCS ======")
    for i, doc in enumerate(docs):
        print(f"\n--- DOC {i} ---")
        print("CONTENT:", doc.page_content)
        print("METADATA:", doc.metadata)
    print("============================\n")
    return docs

def llmchain(vector_db_index, question):
    os.environ["OPENAI_API_KEY"] = open_ai_key

    embedding = OpenAIEmbeddings()

    vector_db_index = vector_db_index
    #vector_db_index = FAISS.load_local(
    #    "vector_db_index",
    #    embeddings=embedding,
    #    allow_dangerous_deserialization=True
    #)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="Using only the following conext {context}. Answer {question}"
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.6,
        max_completion_tokens=500
    )

    retriever = vector_db_index.as_retriever(search_kwargs={"k" : 4})

    chain = (
        {
            "context" : RunnableLambda(lambda x : x["question"]) | retriever | debug_docs,
            "question" : RunnablePassthrough()
        } | prompt | llm | StrOutputParser()
    )

    results = chain.invoke({
        "question" : question
    })
    
    return (results)