"""
Return the concatenated text of all pages in the given PDF file.

The PDF is opened with PyMuPDF, each page's text is extracted with
page.get_text(), and the results are joined with newline characters.

#Assumptions
- did not take into account scanned pdfs
- the current workflow handles normal pdfs and text only in pdf format 
"""

from pathlib import Path
import pymupdf  # PyMuPDF


def extract_pdf_text(path: Path) -> str:
    doc = pymupdf.open(path)
    parts = []
    for page in doc:
        parts.append(page.get_text() or "")
    doc.close()
    return "\n".join(parts).strip()


pdf_path = Path(
    "/Users/abigaelmogusu/projects/LangChain-Projects/medical-rag-service/data/fake-aps.pdf")

# all_text = extract_pdf_text(pdf_path)
# print(all_text[:300])
