from source_data import load_data, split_data, embed_data
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from llm import get_llm


loaded_documents = load_data()
document_chunks = split_data(loaded_documents, chunk_size=1000, chunk_overlap=0)
embedded_documents_searchabledb = embed_data(document_chunks)

retriever = embedded_documents_searchabledb.as_retriever(
    search_kwargs={
        "k": 4
    }
)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question using only this context:\n\n{context}"),
    ("human", "{input}")
])

llm = get_llm()

document_chain = create_stuff_documents_chain(
    llm,
    prompt
)

# 8. Create the full RAG chain
rag_chain = create_retrieval_chain(
    retriever,
    document_chain
)


result = rag_chain.invoke({
    "input": "What is the company's Mobile Phone Policy`?"
})

print(result["answer"])

