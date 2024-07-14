import ast, json

from langchain.schema import SystemMessage
from langchain.chat_models import ChatOpenAI


class Generator:
    
    def __init__(self, openai_api_key, query, document_info: dict, top_k_results: int = 10, model_name='gpt-3.5-turbo-0125', temperature=0.2, max_tokens=4096):
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=openai_api_key,       
            max_tokens=max_tokens
        )
        
        self.query = query
        self.title = document_info["title"]
        self.abstract = document_info["abstract"]
        self.authors = document_info["authors"]
        self.search_results = document_info["sentences"]
        self.top_k_results = top_k_results
        
    async def run_llm(self):
        text = (
            """Um usuário está trabalhando em uma pesquisa científica e fez a seguinte consulta no sistema de recuperação de informação:"""
            f""" "{self.query}" """
            """A partir da consulta foram recuperados trechos e o resumo do seguinte artigo.\n"""
            """A seguir, será passado o titulo, resumo e em seguida os trechos, todos declarados por colchetes.\n"""
            """\n\n"""
            f"""[TITULO]\n{self.title}"""
            """\n\n"""
            f"""[RESUMO]\n{self.abstract}"""
            """\n\n"""
        )

        for doc in self.search_results[:self.top_k_results]:
            text += f"[TRECHO]\n{doc.page_content}\n\n"

        text += (
            """Trechos Finalizados."""
            """Com essas informações, responda do artigo recuperado em relação a consulta realizada."""
            """Sua resposta será no formato JSON com 3 propriedades que respondem:\n"""
            """1. "sumarizacao": Sumarize o artigo recuperado, trazendo informações relevantes, como o tema de pesquisa, as descobertas, objetivos, etc. (máximo 200 palavras)\n"""
            """2. "objetivos": Sumarize os objetivos do artigo. (máximo 100 palavras)\n"""
            """3. "conclusoes": Sumarize as conclusoes do artigo. (máximo 100 palavras)\n"""
            """4. "relevancia": Qual o grau de relevância do artigo para a revisão de literatura do artigo do usuário (altamente relevante, relevante, pouco relevante ou nada relevante)? Se não for relevante, responda apenas 'Não é relevante para a pesquisa'. Se for, explique a relevância do artigo para a consulta, como seu uso para contribuir com a pesquisa do usuário. (máximo 200 palavras)\n"""
        )
        
        try:
            messages = [SystemMessage(content=text)]
            response = self.llm(messages)
            content = ast.literal_eval(response.content)
        except Exception as e:
            raise(e)
        
        self.llm_query = text
        self.llm_summary = content["sumarizacao"]
        self.llm_objectives = content["objetivos"]
        self.llm_conclusions = content["conclusoes"]
        self.llm_relevance = content["relevancia"]

    @property
    def article_summary(self):
        if not hasattr(self, "llm_summary"):
            raise Exception("Summary not found. First execute .run_llm() method.")
        return self.llm_summary

    @property
    def article_objectives(self):
        if not hasattr(self, "llm_objectives"):
            raise Exception("Objectives not found. First execute .run_llm() method.")
        return self.llm_objectives

    @property
    def article_conclusions(self):
        if not hasattr(self, "llm_conclusions"):
            raise Exception("Conclusions not found. First execute .run_llm() method.")
        return self.llm_conclusions

    @property
    def article_relevance(self):
        if not hasattr(self, "llm_relevance"):
            raise Exception("Relevance not found. First execute .run_llm() method.")
        return self.llm_relevance
