import os
import json

## Own Lib ##

from lib.retriever import Retriever
from lib.reranker import Reranker
from lib.generator import Generator


## METADATA ##

pdf_folder_path = "./metadata_documents"

filename_list = []
for filename in os.listdir(pdf_folder_path):
    if filename.endswith(".json") and not filename.endswith("_x.json"):
        filename_list.append(os.path.join(pdf_folder_path, filename))

filename_list = sorted(filename_list)

def get_metadata(filename_list):
    metadata_dict = {}
    
    for filename in filename_list:
        with open(filename, "r") as file: 
            metadata = json.load(file)
            _id = metadata["id"]
            metadata_dict[_id] = metadata
    
    return metadata_dict

metadata_dict = get_metadata(filename_list)


## VECTOR STORE ##

from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings

print("Recuperando Embeddings...")
embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

print("Construindo Vector Database...")
vectorstore = FAISS.load_local("./faiss_vector_store/", embeddings)


## OTHERS ##
    
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


## FULL RETRIEVE-RERANK-GENERATE ##

import math

def print_result(document, generator):
    print("\n", "-" * 20, "\n")

    stars = "*" * math.ceil(document['relevance_score'] / 0.20)
    print(color.BOLD + color.RED + f"[{document['id']}] [{stars}] {document['title']}" + color.END)

    print(color.BOLD + color.BLUE + f" Relevância para a Pesquisa: {round(document['relevance_score'] * 100, 2)}%" + color.END)

    print("\n", "; ".join(document['authors']))
    
    print(color.BOLD + "\n Relevância:" + color.END, 
        generator.article_relevance)
    print(color.BOLD + "\n Resumo:" + color.END,
        generator.article_summary)
    print(color.BOLD + "\n Objetivos:" + color.END,
        generator.article_objectives)
    print(color.BOLD + "\n Conclusões:" + color.END,
        generator.article_conclusions)
    

import asyncio

openai_api_key = "xxxxx"
cohere_key = "xxxxx"

loop = asyncio.new_event_loop()


print("Construindo Objetos...")

retriever = Retriever(vectorstore)
reranker = Reranker(cohere_key=cohere_key, model="rerank-multilingual-v2.0")


print("\n-- Sistema Iniciado --\n")

query = input("Escreva sobre seu tema: ")
while query is not None:
    os.system('cls' if os.name == 'nt' else 'clear')
    print("Consulta Realizada:", query)
    
    print("\nRecuperando Documentos Similares...")

    retrieved_documents = loop.run_until_complete(
        retriever.search(
            query, k=100, fetch_k=10000, max_k_by_document=4, filter={"type":"sentences"}
        )
    )

    # print("Retrieved Documents:", list(retrieved_documents.keys()))
    print("Filtrando Documentos Recuperados...")

    documents = reranker.prepare_documents(metadata_dict, retrieved_documents)
    reranked_documents = reranker.rerank(query, documents, 5, minimum_relevance=0.3)

    print("Gerando sumarizações...")

    for doc in reranked_documents:
        generator = Generator(openai_api_key, query, doc)
        loop.run_until_complete(generator.run_llm())

        print_result(doc, generator)


    print("\n", "-" * 20, "\n")
    query = input("Escreva sobre seu tema: ")

