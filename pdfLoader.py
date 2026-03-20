import fitz  # PyMuPDF

class PDFLoader:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def extract_text(self):
        doc = fitz.open(self.pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

# if __name__ == '__main__':
    # loader = PDFLoader('contextPDF.pdf')
    # text = loader.extract_text()
    # print(text)