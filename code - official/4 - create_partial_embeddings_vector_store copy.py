import os
import json

from langchain.vectorstores import FAISS

from langchain.embeddings import SentenceTransformerEmbeddings

embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")



# == QUEBRANDO ARQUIVOS == # 

pdf_folder_path = "./formatted_documents"

filename_list = []
for filename in os.listdir(pdf_folder_path):
    if filename.endswith(".json"):
        filename_list.append(os.path.join(pdf_folder_path, filename))

filename_list = sorted(filename_list)

def retrieve_document(filename):
    with open(filename, "r") as file:
        json_file = json.load(file)
    return json_file


from lib.document import DocumentConstructor
    
# MAX WINDOW SIZE (quantidade de paragrafos sequencias unidos)
# MAX TOKEN SIZE (quantidade de tokens em um embedding)
# MIN TOKEN SIZE (quantidade minima de tokens em um embedding)
# MIN_SIZE_DIFFERENCE (quantidade minima de tokens entre dois embeddings de windows - evitar duplicatas)

# document_constructor = DocumentConstructor(2, 500, 100, 0)    # antigo
document_constructor = DocumentConstructor(5, 500, 150, 200)
    
total_files = len(filename_list)
print("Total Files:", total_files)

total_documents = 0
for start in range(0, total_files, 100):
    end = start + 100
    
    print("Starting", start, "...", end)
    docs = []
    for filename in filename_list[start:end]:
        json_doc = retrieve_document(filename)
        doc = document_constructor.create_documents(json_doc)
        docs.extend(doc)

        print('.', end='')
    
    total_documents += len(docs)
    print(len(docs), "\n")
    
    if not os.path.exists(f"./partial_faiss_vector_store_{start}/"):
        os.makedirs(f"./partial_faiss_vector_store_{start}/")
    
    new_vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    new_vectorstore.save_local(f"./partial_faiss_vector_store_{start}/")

print("Total Documents:", total_documents)

