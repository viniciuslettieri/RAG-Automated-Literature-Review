from langchain.docstore.document import Document

class DocumentConstructor:
    def __init__(self, MAX_WINDOW_SIZE, MAX_TOKEN_SIZE, MIN_TOKEN_SIZE, MIN_SIZE_DIFFERENCE):
        self.MAX_WINDOW_SIZE = MAX_WINDOW_SIZE
        self.MAX_TOKEN_SIZE = MAX_TOKEN_SIZE
        self.MIN_TOKEN_SIZE = MIN_TOKEN_SIZE
        self.MIN_SIZE_DIFFERENCE = MIN_SIZE_DIFFERENCE

    def create_title_document(self, title, document_id):
        doc = Document(page_content=title, metadata={"id": document_id, "type": "title"})
        return doc

    def create_abstract_document(self, abstract, document_id):
        doc = Document(page_content=abstract, metadata={"id": document_id, "type": "abstract"})
        return doc
    
    def create_sentence_document(self, text, document_id, start_index, window_size):
        doc = Document(page_content=text, metadata={"id": document_id, "type": "sentences", "start_index": start_index, "window_size": window_size})
        return doc

    def create_documents(self, json_document):
        docs = []

        title_doc = self.create_title_document(json_document["title"], json_document["id"])
        docs.append(title_doc)

        abstract_doc = self.create_abstract_document(json_document["abstract"], json_document["id"])
        docs.append(abstract_doc)

        sentences = json_document["sentences"]
        sentences_lens = [len(s.split()) for s in sentences]
        
        for index in range(len(sentences)):
            saved_length = None
            for window_size in range(1,self.MAX_WINDOW_SIZE+1):
                sentence_list = sentences[index:index+window_size]
                sentence_list_lens = sentences_lens[index:index+window_size]

                sum_s = sum(sentence_list_lens)
                if (window_size < self.MAX_WINDOW_SIZE and sum_s < self.MIN_TOKEN_SIZE) or (window_size > 1 and sum_s > self.MAX_TOKEN_SIZE):
                    continue
                
                # Only caches if the difference is sufficient
                if saved_length is None or abs(sum_s - saved_length) > self.MIN_SIZE_DIFFERENCE:
                    text = '\n'.join(sentence_list)
                    sentence_doc = self.create_sentence_document(text, json_document["id"], index, window_size)
                    docs.append(sentence_doc)
                    saved_length = sum_s
                
        return docs