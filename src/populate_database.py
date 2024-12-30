import argparse # default recommended standard library module for implementing basic command line applications.
import os       # library provides a portable way of using operating system dependent functionality.
import shutil   # library forhigh-level operations like copying and removal on collections of files.
import common   # Import common dependancy functions

from langchain_community.document_loaders import PyPDFDirectoryLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma

#main function
def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    # Get database paths for each category
    db_paths = common.get_db_paths()
    
    # Clear existing databases
    for db_path in db_paths.values():
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
    
    # Load documents by category
    documents_by_category = load_documents_by_category(common.DATA_PATH)
    
    # Process each category separately
    for category, documents in documents_by_category.items():
        if not documents:
            continue
            
        # Split documents into chunks
        chunks = split_documents(documents)
        
        # Add chunks to category-specific database
        add_to_chroma(chunks, db_paths[category])

def load_documents():
    documents = []
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(common.DATA_PATH):
        # Load PDFs from current directory
        pdf_dir = [f for f in files if f.lower().endswith('.pdf')]
        if pdf_dir:
            pdf_path = os.path.join(root)
            documents.extend(PyPDFDirectoryLoader(pdf_path).load())
        
        # Load TXT files from current directory
        txt_files = [f for f in files if f.lower().endswith('.txt')]
        for file in txt_files:
            file_path = os.path.join(root, file)
            documents.extend(UnstructuredFileLoader(file_path).load())
            
    return documents


# populate_database.py modifications
def load_documents_by_category(data_path):
    """
    Load documents organized by top-level category
    Returns a dictionary with category names as keys and document lists as values
    """
    documents_by_category = {}
    
    for root, dirs, files in os.walk(data_path):
        depth = root[len(data_path):].count(os.sep)
        if depth > 2:  # Skip directories deeper than level 2
            continue
            
        # Get the top-level directory name
        rel_path = os.path.relpath(root, data_path)
        top_dir = rel_path.split(os.sep)[0]
        
        if top_dir not in documents_by_category:
            documents_by_category[top_dir] = []
            
        # Load PDFs
        pdf_files = [f for f in files if f.lower().endswith('.pdf')]
        if pdf_files:
            documents_by_category[top_dir].extend(PyPDFDirectoryLoader(root).load())
            
        # Load TXT files
        txt_files = [f for f in files if f.lower().endswith('.txt')]
        for file in txt_files:
            file_path = os.path.join(root, file)
            documents_by_category[top_dir].extend(UnstructuredFileLoader(file_path).load())
    
    return documents_by_category

def split_documents(documents):
    """
    Split documents based on their type with improved handling.
    """
    pdf_docs = []
    txt_docs = []
    
    # Separate documents by type
    for doc in documents:
        source = doc.metadata.get('source', '').lower()
        if source.endswith('.pdf'):
            pdf_docs.append(doc)
        elif source.endswith('.txt'):
            txt_docs.append(doc)
    
    # Split PDFs by pages
    pdf_chunks = split_pdfs_by_page(pdf_docs)
    
    # Split TXT files by sentences with improved handling
    txt_chunks = split_txt_by_sentences(txt_docs)
    
    # Return combined chunks
    return pdf_chunks + txt_chunks

def split_pdfs_by_page(pdf_docs):
    chunks = []
    for doc in pdf_docs:
        # Get the page number from metadata
        page_num = doc.metadata.get('page', 0)
        
        # Create a new chunk for the entire page
        chunks.append(Document(
            page_content=doc.page_content,
            metadata={
                'source': doc.metadata.get('source'),
                'page': page_num
            }
        ))
    return chunks

def split_txt_by_sentences(txt_docs):
    """
    Split text documents into proper sentences with improved handling.
    Uses a combination of RecursiveCharacterTextSplitter for initial chunking
    and additional sentence-level processing.
    """
    # Initialize text splitter with more appropriate settings
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=common.CHUNK_SIZE,  # Use the configured chunk size
        chunk_overlap=common.CHUNK_OVERLAP,  # Use the configured overlap
        length_function=len,
        is_separator_regex=False,
    )
    chunks = []
    for doc in txt_docs:
        # Get basic document metadata
        source = doc.metadata.get('source', '')
        page = doc.metadata.get('page', 1)
        
        # First, split into manageable chunks
        splits = text_splitter.split_text(doc.page_content)
        
        # Process each chunk to ensure proper sentence boundaries
        for chunk in splits:
            # Create Document object for the chunk
            chunks.append(Document(
                page_content=chunk,
                metadata={
                    'source': source,
                    'page': page
                }
            ))
    
    return chunks



#function to add the chunks into the chroma vector database
def add_to_chroma(chunks, db_path):
    """
    Add document chunks to Chroma database, handling duplicates
    """
    # Load the database
    db = Chroma(persist_directory=db_path, embedding_function=common.embedding_function())
    
    # Calculate chunk IDs
    chunks_with_ids = calculate_chunk_ids(chunks)
    
    # Get existing IDs
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    
    # Filter out duplicates
    new_chunks = []
    new_chunk_ids = []
    seen_ids = set()
    
    for chunk in chunks_with_ids:
        chunk_id = chunk.metadata["id"]
        if chunk_id not in existing_ids and chunk_id not in seen_ids:
            new_chunks.append(chunk)
            new_chunk_ids.append(chunk_id)
            seen_ids.add(chunk_id)
    
    if new_chunks:
        print(f"👉 Adding new documents to {os.path.basename(db_path)}: {len(new_chunks)}")
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print(f"✅ No new documents to add to {os.path.basename(db_path)}")


def calculate_chunk_ids(chunks):
    """
    Calculate unique IDs for document chunks based on source file, page number, and chunk index.
    
    Args:
        chunks: List of Document objects
        
    Returns:
        List of Document objects with unique IDs added to metadata
    """
    # Track chunk indices per page
    page_indices = {}
    
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page", 0)
        
        # Create unique page identifier
        page_id = f"{source}:{page}"
        
        # Initialize or increment chunk index for this page
        if page_id not in page_indices:
            page_indices[page_id] = 0
        else:
            page_indices[page_id] += 1
            
        # Calculate unique chunk ID
        chunk_id = f"{page_id}:{page_indices[page_id]}"
        
        # Add to chunk metadata
        chunk.metadata["id"] = chunk_id
        
    return chunks

if __name__ == "__main__":
    main()
