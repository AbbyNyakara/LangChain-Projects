'''
Docstring for src.chunking.semantic

The code splits the generated data semantically in a slow, but meaning-aware manner, resulting in chunks
that preserve meaning and structure: - P.S- Doesnt work for some reason?

Fallback - Used Recursive Character Text splitter - Chunks, tries to preserve the semantic meaning 

'''


#  from langchain_text_splitters import SemanticChunker
#  from langchain_experimental.text_splitters import SemanticChunker

'''
def chunk_text_semantic(text):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    splitter = SemanticChunker(embeddings)
    docs = splitter.create_documents([text])
    return [doc.page_content for doc in docs]

'''

from langchain_text_splitters import RecursiveCharacterTextSplitter
from document.extractor import extract_pdf_text
import sys
from pathlib import Path

src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

# NOW import from document


def chunk_text(text: str) -> list[str]:
    """Split text into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    return chunks
