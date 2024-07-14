# Automated Literature Review with Retrieval-Augmented Generation (RAG)

Repository for my final undergraduate project, focused on studying and applying the Retrieval-Augmented Generation paradigm in automating the bibliographic review process in Literature Reviews, with the goal of making research work more efficient.

![arquitetura](https://github.com/user-attachments/assets/2f1d3335-c5b5-43e1-a136-6bb4dfd05e78)

# Notebooks and Programs

The official code is saved in 'code - official'.

For the preprocessing part, from data acquisition to the vector store.

1. **Metadata Parsing**: From the metadata information downloaded from "Base Minerva" in RIS Format (saved in ./metadata_minerva), parse the RIS format to Json format with title, authors, abstract, bibliography pages, keywords, urls, pdf urls, and the id. The parsed data is saved to ./metadata_documents.

2. **PDF Retrieval**: From the parsed metadata files, retrieves PDF files for the underlying researches and parses them with PyPDF2 reader. For each document the text is joined into one continuous page. All parsed PDF text files are saved to ./parsed_pdf.

3. **Text Parsing**: The text extract from the PDF can have formatting issues, treated by the parsing step. This step identifies text continuity and end of lines. Also ignores possible header and footer text. Every document is formatted and saved to ./formatted_documents.

4. **Paragraph Chunking and Embedding**: This part could not be completed with jupyter notebooks because of memory issues. The files are simple python programs divided into 'partial' and 'final'. The partial gets each formatted document and runs the DocumentConstructor from the lib to retrieve the chunks with: max window n-gram size (5), max token total (500), min token total (150) and min difference in size between n-grams (200). This returns the document chunks that will be partially added into partial FAISS vector store 100 by 100. Those partial vector store need to be joined together by the other code. This solved the memory overhead issue. The final faiss vector store will be stored to ./faiss_vector_store.


For the Real Time test of the system:

5. **Terminal App**: Retrieves and runs the search and summarization system real time from the user input.


For the validation part:

6. **Preprocessing for the Validation**: Retrieves information from all the documents and creates tests to be executed. The tests are saved to be executed without randomness. For each document, the objectives are extracted with an LLM. This will be inputed to the system to validate it. The tests information are stored to ./validation_tests.

7. **Validation Tests**: The tests are executed with the later information. From the query generated, the retriever and generator are triggered and the results are saved to calculate metrics.


# Lib Files

1. **Text Parser**: From a continuous text string, parses into paragraphs.

2. **Document Constructor**: Receives a document from ./formatted_documents and creates document chunks for the texts.

3. **Retriever**: Retrieves from the vector store the top ranked chunks grouped by the original document.

4. **Reranker**: Reranks a list of grouped documents by relevance to the query.
   
5. **Generator**: Generates the summaries for: document summary, document objectives, document conclusions, document relevance to the query.
