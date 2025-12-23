from pathlib import Path
import pymupdf  # PyMuPDF

class PDFExtractor:
    def __init__(self, pdf_path: Path = "/Users/abigaelmogusu/projects/LangChain-Projects/medical-rag-service/data/fake-aps.pdf"):
        self.pdf_path = pdf_path


    def extract(self) -> str:
        """Extract text from PDF and return as string."""
        doc = pymupdf.open(self.pdf_path)
        parts = []
        for page in doc:
            parts.append(page.get_text() or "")
        doc.close()
        return "\n".join(parts).strip()

# # Usage
# pdf_path = Path(
#     "/Users/abigaelmogusu/projects/LangChain-Projects/medical-rag-service/data/fake-aps.pdf"
# )
# extractor = PDFExtractor(pdf_path)
# all_text = extractor.extract()
# print(all_text[:300])
