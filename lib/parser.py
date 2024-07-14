import re
import numpy as np
from unidecode import unidecode


class TextParser:
    
    TOKEN_CTX_SIZE = 1000
    
    def __init__(self, DEBUG_MODE=False):
        self.DEBUG_MODE = DEBUG_MODE

    def parse_paragraph(self, text: str) -> str:
        parsed = re.sub(r'[ \t]+', ' ', text)
        parsed = parsed.strip()        
        return parsed

    def should_ignore_line(self, line: str) -> bool:
        line = line.strip()
        line_size = len(line)

        if line == '' or len(line) == 0:
            if self.DEBUG_MODE: print("Removed by 'Empty Line':", line)
            return True
        
        if line.isdigit():
            if self.DEBUG_MODE: print("Removed by 'Is Digit':", line)
            return True

        line_upper = unidecode(line.upper().replace(" ", ""))
        
        # sumario
        match_summary_start = re.search(r'^[0-9\.]+[A-Z]+', line_upper)
        match_summary_end = re.search(r'[A-Z\.]+[0-9]+$', line_upper)
        if line_upper.count('.') > 0.2 * line_size or (match_summary_start is not None and match_summary_end is not None):
            if self.DEBUG_MODE: print("Removed by 'Is Summary':", line)
            return True
        
        # poucas letras
        count_letters = sum([line_upper.count(letter) for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'])
        if count_letters / len(line_upper) < 0.5:
            if self.DEBUG_MODE: print("Removed by 'Little Text':", line)
            return True

        # figuras
        match_figura = re.search(r'^FIGURA[0-9]+', line_upper)
        if match_figura is not None:
            if self.DEBUG_MODE: print("Removed by 'Is Figure':", line)
            return True

        # tabelas
        match_tabela = re.search(r'^TABELA[0-9]+', line_upper)
        if match_tabela is not None:
            if self.DEBUG_MODE: print("Removed by 'Is Table':", line)
            return True

        # tabelas
        match_quadro = re.search(r'^QUADRO[0-9]+', line_upper)
        if match_quadro is not None:
            if self.DEBUG_MODE: print("Removed by 'Is Quadro':", line)
            return True
        
        # anexo
        match_anexo = re.search(r'^ANEXO[0-9]+', line_upper)
        if match_anexo is not None:
            if self.DEBUG_MODE: print("Removed by 'Is Attachment':", line)
            return True
            
        return False

    def is_end_of_paragraph(self, line: str) -> bool:
        # here we define all the ponctuations that may end a paragraph
        match_end = re.search(r'[\.\:\!\?][ \t]*$', line)
        return (match_end is not None)
    
    def is_title(self, line: str) -> bool:
        line_upper = unidecode(line.upper().replace(" ", ""))
        match_title = re.search(r'^[0-9\.]+[A-Z]+', line_upper)
        return (match_title is not None)
    
    def parse_block(self, text: str) -> list:
        """
        Parses a block of text that may contain multiple paragraphs.
        The purpose is to identify and break the paragraphs, and exclude unnecessary lines.
        """
        
        lines = text.split('\n')
        lines_len = [len(line) for line in lines]
        
        mean = np.mean(lines_len)
        std = np.std(lines_len)
        inferior_confidence_interval = mean - 1.65 * std / np.sqrt(len(lines))
    
        paragraphs = []
        joined_text = ""
        for line in lines:           
            if self.should_ignore_line(line):
                continue
    
            end_of_paragraph = self.is_end_of_paragraph(line)
            is_title = self.is_title(line)
            if end_of_paragraph and len(line) < inferior_confidence_interval:
                if self.DEBUG_MODE: print("End of Paragraph:", line)
                
                joined_text += line
                parsed_paragraph = self.parse_paragraph(joined_text)
                paragraphs.append(parsed_paragraph)
                joined_text = ""   
                
            elif is_title and len(line) < inferior_confidence_interval:
                if self.DEBUG_MODE: print("Is Title:", line)
                
                parsed_paragraph = self.parse_paragraph(joined_text)
                paragraphs.append(parsed_paragraph)
                joined_text = ""
                
                parsed_paragraph = self.parse_paragraph(line)
                paragraphs.append(parsed_paragraph)
                
            else:
                joined_text += line + " "
                
    
        if joined_text != "":
            parsed_paragraph = self.parse_paragraph(joined_text)
            paragraphs.append(parsed_paragraph)

        final_filtered = [p for p in paragraphs if not self.should_ignore_line(p)]        
        return final_filtered

    def limit_paragraph_tokens(self, text: str) -> list:
        text_split = text.split()
        if len(text_split) < self.TOKEN_CTX_SIZE:
            return [text]
        else:
            return text.split('.')

    def filter_pages(self, text_pieces):
        """Filters out everything before the abstract and after bibliography"""

        total_paragraphs = len(text_pieces)
        total_length_paragraphs = np.cumsum([len(p) for p in text_pieces])
        initial = total_paragraphs
        end = 0
        
        for idx, p in enumerate(text_pieces):
            text = unidecode(p.upper().replace(" ", ""))
            
            has_resumo = "RESUMO" in text
            has_abstract = "ABSTRACT" in text
            
            has_referencia = "REFERENCIA" in text
            has_bibliograf = "BIBLIOGRAF" in text
            at_end = total_length_paragraphs[idx] > 0.75 * total_length_paragraphs[-1]
            small_size = len(text) < 30
            
            if (has_resumo or has_abstract):
                initial = min(initial, idx)
            if (has_referencia or has_bibliograf) and at_end and small_size:
                end = max(end, idx)
        
        if initial == total_paragraphs:
            initial = 0
        if end == 0:
            end = total_paragraphs
            
        if self.DEBUG_MODE: print(f"Filtered Text starting at {initial}/{total_paragraphs} and ending at {end}/{total_paragraphs}")

        filtered_pieces = text_pieces[initial:end]
        return filtered_pieces
    
    def parse_full_text(self, full_text: str):
        parsed_texts = []
        
        for text in re.split(r'\n[ \t]+\n', full_text):
            blocks = self.parse_block(text)
            for block in blocks:
                limited_blocks = self.limit_paragraph_tokens(block)
                parsed_texts.extend(limited_blocks)

        filtered_text = self.filter_pages(parsed_texts)

        return filtered_text