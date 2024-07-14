import os

from time import sleep

from langchain.vectorstores import FAISS

from langchain.embeddings import SentenceTransformerEmbeddings

embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")


# == UNINDO ARQUIVOS == #

filename_list = []
for filename in os.listdir('./'):
    if filename.startswith("partial_faiss_vector_store_"):
        print(filename)
        filename_list.append(os.path.join('./', filename))

total_files = len(filename_list)
batch = 8


for start in range(0, total_files, batch):
    end = start + batch
    
    print("Starting", start, "...", end)

    vectorstore = FAISS.load_local(filename_list[start], embeddings=embeddings)
    for filename in filename_list[start+1:end+1]:
        new_vectorstore = FAISS.load_local(filename, embeddings=embeddings)
        vectorstore.merge_from(new_vectorstore)
        print(filename)
        del(new_vectorstore)
        sleep(5)
    
    if not os.path.exists(f"./combined_faiss_vector_store_{start}/"):
        os.makedirs(f"./combined_faiss_vector_store_{start}/")
    
    vectorstore.save_local(f"./combined_faiss_vector_store_{start}/")
    print(f"./combined_faiss_vector_store_{start}/")
    del(vectorstore)
    sleep(5)



# == UNINDO ARQUIVOS 2 == #

filename_list = []
for filename in os.listdir('./partial_faiss_vector_store3'):
    if filename.startswith("partial_faiss_vector_store_"):
        print(filename)
        filename_list.append(os.path.join('./partial_faiss_vector_store3/', filename))

vectorstore = FAISS.load_local(filename_list[0], embeddings=embeddings)
print(filename_list[0])
for filename in filename_list[1:]:
    new_vectorstore = FAISS.load_local(filename, embeddings=embeddings)
    vectorstore.merge_from(new_vectorstore)
    print(filename)
    del(new_vectorstore)
    sleep(30)

if not os.path.exists(f"./new_faiss_vector_store/"):
    os.makedirs(f"./new_faiss_vector_store/")

vectorstore.save_local(f"./new_faiss_vector_store/")
print(f"./new_faiss_vector_store/")
del(vectorstore)
sleep(5)



# == UNINDO ARQUIVOS COMBINADOS == #

# part1 = FAISS.load_local(f"./combined_faiss_vector_store_0/", embeddings=embeddings)
# part2 = FAISS.load_local(f"./combined_faiss_vector_store_8/", embeddings=embeddings)
# part1.merge_from(part2).save_local(f"./faiss_vector_store/")
