
import sys
from pathlib import Path
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from extractor.extractor import PDFExtractor

class TextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk(self, text: str) -> list[str]:
        return self.splitter.split_text(text)

# Usage
# extractor = PDFExtractor()
# extracted_text = extractor.extract()
# chunker = TextChunker() 
# chunks = chunker.chunk(extracted_text)

# print(f"Created {len(chunks)} chunks")

