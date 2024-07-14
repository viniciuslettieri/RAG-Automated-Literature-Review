import cohere
from copy import deepcopy

class Reranker:
    
    def __init__(self, cohere_key, model):
        self.cohere_client = cohere.Client(cohere_key)
        self.model = model

    def prepare_documents(self, metadata_dict: dict, retrieved_documents: dict):
        """
        - metadata_dict: all metadata
        - retrieved_documents: retrieved sentence documents grouped by key
        """

        retrieved_documents = deepcopy(retrieved_documents)
        
        documents = {}
        for key in retrieved_documents:
            documents[key] = metadata_dict[key]
            documents[key]["sentences"] = [doc[1] for doc in retrieved_documents[key]]      # score and doc tuple
        
        return documents
        
    def encode_document(self, document: dict, using: list):
        sentences_text = [doc.page_content for doc in document["sentences"]]
        sentences = "\n\n".join(sentences_text)
        
        text = ""
        if 'title' in using: text += document['title'] + "\n\n"
        if 'abstract' in using: text += document['abstract'] + "\n\n"
        if 'sentences' in using: text += sentences + "\n\n"
        
        return text
    
    def rerank(self, query: str, documents: dict, top_n: int, using: list = ["title", "abstract", "sentences"], minimum_relevance=0.5):
        """
        - query: string used by to retrieve documents
        - documents: dictionary with key as the article id and value as the metadata for the document, with title, abstract and sentences
        - top_n: number of top documents to return
        - using: list of elements to include in documents
        - return: list of document metadata
        """

        documents = deepcopy(documents)
        
        keys_list = list(documents.keys())
        document_list = [self.encode_document(documents[key], using) for key in keys_list]

        response = self.cohere_client.rerank(
            model = self.model,
            query = query,
            documents = document_list,
            top_n = top_n
        )
        
        ordered_documents = []
        for res in response:
            key = keys_list[res.index]
            doc = documents[key]
            doc["relevance_score"] = res.relevance_score

            if res.relevance_score >= minimum_relevance:
                ordered_documents.append(doc)
        
        return ordered_documents
