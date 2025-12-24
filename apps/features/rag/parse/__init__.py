from .process_pdf import PdfParser
from .process_excel import ExcelParser
from .process_word import WordParser

excel_parser = ExcelParser()
word_parser = WordParser()
pdf_parser = PdfParser()
__all__ = [
    'excel_parser',
    'word_parser',
    'pdf_parser'
]
