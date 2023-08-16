import timeit
from optimized_data_pro import data
from langchain.schema import Document
processed_data = []
for item in data:
    if item.get("details")!= None:
        processed_item = Document(
            page_content="I want to buy the "+item["brand"]+"'s "+ item["similarityWith"]+" with product model number " + item["details"]["model_number"]+" and price "+str(item["details"]["price"])+" provided the discount price of "+str(item["details"]["discount_price"]) + " and emi of "+ str(item["details"]["emi_price"])+".",
            meta = item
        )
        processed_data.append(processed_item)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
texts = text_splitter.split_documents(processed_data)
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                   model_kwargs={'device': 'cuda'})
vectorstore = FAISS.from_documents(texts, embeddings)
vectorstore.save_local('vectorstore/db_faiss')

qa_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Improvise if necessary.
Context: {context}
Question: {question}
Only return the helpful answer below and nothing else.
Helpful answer:
"""

from langchain.llms import CTransformers

# Local CTransformers wrapper for Llama-2-7B-Chat
llm = CTransformers(model='/content/drive/MyDrive/models/llama-2-7b-chat.ggmlv3.q8_0.bin', # Location of downloaded GGML model
                    model_type='llama', # Model type Llama
                    config={'max_new_tokens': 256,
                            'temperature': 0.01},
                    device='cuda')

from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Wrap prompt template in a PromptTemplate object
def set_qa_prompt():
    prompt = PromptTemplate(template=qa_template,
                            input_variables=['context', 'question'])
    return prompt


# Build RetrievalQA object
def build_retrieval_qa(llm, prompt, vectordb):
    dbqa = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=vectordb.as_retriever(search_kwargs={'k':2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt})
    return dbqa


# Instantiate QA object
def setup_dbqa():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.load_local('vectorstore/db_faiss', embeddings)
    qa_prompt = set_qa_prompt()
    dbqa = build_retrieval_qa(llm, qa_prompt, vectordb)
    return dbqa


def llmresp(query):
    start = timeit.default_timer()  # Start timer
    input=query
    dbqa = setup_dbqa()  # Define or import setup_dbqa() earlier in the notebook
    response = dbqa({'query': input})
    print(type(response))
    print(response)
    end = timeit.default_timer()  # End timer
    print(f'\nAnswer: {response["result"]}')
    print('=' * 50)  # Formatting separator
    source_docs = response['source_documents']
    for i, doc in enumerate(source_docs):
        print(f'\nSource Document {i + 1}\n')
        print(f'Source Text: {doc.page_content}')
    print(f"Time to retrieve response: {end - start}")
    return source_docs

from typing import Union
import os
import timeit
from fastapi import FastAPI
app = FastAPI()
from haystack.document_stores import InMemoryDocumentStore
from haystack import Document
from haystack.nodes import BM25Retriever
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline

def process_api_data(sub):
    processed_data = []
    for item in sub:
        if item.get("details")!= None:
            processed_item = Document(
                content="I want to buy the "+item["brand"]+"'s "+ item["similarityWith"]+" with product model number " + item["details"]["model_number"]+" and price "+str(item["details"]["price"]),
                meta=item
            )
            processed_data.append(processed_item)
    return processed_data
processed_api_data = process_api_data(data)
document_store = InMemoryDocumentStore(use_bm25=True)
document_store.write_documents(processed_api_data)
retriever = BM25Retriever(document_store=document_store)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
pipe = ExtractiveQAPipeline(reader, retriever)

@app.get("/")
def parse_query(query: str or None):
    input=query
    prediction = pipe.run(
    query="The price of "+query,
    params={"Retriever": {"top_k": 40}, "Reader": {"top_k": 30}}
    )
    print(query)
    dic={"products":[]
    }
    for ans in prediction["answers"]:
        print(ans)

        if ans.score >= 0.05:
            if ans.meta["type"]=="products":
                dic["products"].append({
                    'meta':ans.meta
                })
    ans=llmresp(input)
    try:
        best_recommendation=dic["products"][0]
        response={
            "Answer": ans,
            "title": "This is the recommended products",
            "type": "recommend-slider",
            "subtitle": "Recommendations",
            "detailsIframe": {
            "heading": "Similar Products",
            'data':[],
            "link":''
            },
        }
        products=[]
        if len(dic['products'])>0:
            print(dic['products'][0],"====product=======")
            details=dic['products'][0].get('meta')
            link_name=details.get("name").lower().replace('/','').replace('"','').replace(" ",'-')
            product_link=f"https://cgdigital.com.np/product-{link_name}-{details.get('id')}?botIframe=true&removeByTag=header,footer,Header&removeById=breadcrumb-wrapper&removeByClassName=sort-wrp"
            response['detailsIframe']['link']=product_link
            for catagories in dic['products']:
                details=catagories.get('meta').get('details')
                products.append({
                        "title": details.get('name'),
                        "image": details.get('featured_image'),
                        "price": details.get('price'),
                        "iframe":product_link,
                        "model_number": details.get('model_number'),
                        "quantity": details.get('quantity'),
                        "availability": details.get('availability'),
                        "brand": details.get('brand')
                })
            response['data']=products

        return response
    except IndexError as e:
        return {"message": "Product not available"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}