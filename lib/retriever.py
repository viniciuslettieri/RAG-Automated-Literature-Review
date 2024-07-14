class Retriever:
    
    def __init__(self, vectorstore, MINIMUM_RESULTS_PER_DOCUMENT=3):
        self.vectorstore = vectorstore
        self.MINIMUM_RESULTS_PER_DOCUMENT = MINIMUM_RESULTS_PER_DOCUMENT
    
    def _group_documents(self, retrieved_documents: list):
        """Groups documents by id"""
        
        results = {}

        # geramos um mapa com id.start_index.[lista dos documents que iniciam naquele index]
        for doc in retrieved_documents:
            _id = doc[0].metadata["id"]
            _start_index = doc[0].metadata.get("start_index") or 0

            if _id not in results.keys():
                results[_id] = {}

            if _start_index not in results[_id].keys():
                results[_id][_start_index] = []

            results[_id][_start_index].append(doc)
        
        return results
    
    def _organize_documents(self, grouped_documents: dict):
        """
        Organizes documents by better similarity results.
        Chooses longest document when multiple start on the same paragraph.
        Ignores documents with less than 2 results.
        """

        results_by_document = {}
        for _id, docs in grouped_documents.items():

            if len(docs) < self.MINIMUM_RESULTS_PER_DOCUMENT:
                # print("Documento", _id, "ignorado por pouco conteúdo similar.")
                continue

            results_by_document[_id] = []

            for start_index, doc_tuples in docs.items():
                # a partir disso podemos selecionar apenas para cada start_index o de maior janela
                elements = sorted(doc_tuples, key=lambda x: (x[1], -(x[0].metadata.get("window_size") or 1)), reverse=False)
                results_by_document[_id].append((elements[0][1], elements[0][0]))
        
        return results_by_document

    def _limit_k_by_document(self, organized_documents, max_k_by_document):
        organized_documents = organized_documents.copy()
        
        for _id in organized_documents.keys():
            doc_tuples = organized_documents[_id]
            sorted_by_score = sorted(doc_tuples, key=lambda x: x[0], reverse=False)
            top_k_sorted_by_score = sorted_by_score[:max_k_by_document]
            
            organized_documents[_id] = top_k_sorted_by_score
            
        return organized_documents

    async def search(self, query, k, fetch_k, max_k_by_document=100, filter=None):
        res = self.vectorstore.similarity_search_with_score(
            query,
            k=k,
            fetch_k=fetch_k,
            filter=filter
        )
        
        grouped_documents = self._group_documents(res)
        organized_documents = self._organize_documents(grouped_documents)
        top_k_documents = self._limit_k_by_document(organized_documents, max_k_by_document)
        
        return top_k_documents
