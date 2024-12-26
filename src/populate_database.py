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
    # Initialize a new argument parser (class)
    parser = argparse.ArgumentParser()

    # For our parser, add an optional boolean flag for database reset. When present, sets args.reset to True
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    
    # Parse command-line arguments
    args = parser.parse_args()

    # Check if reset flag was provided
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Load list of documents from source
    documents = load_documents()

    # Split documents into chunks, which are instances of class langchain.schema.document.Document
    chunks = split_documents(documents)

    # Add chunks to vector database
    add_to_chroma(chunks)

def load_documents():
    documents = []
    # Load PDFs
    documents.extend(PyPDFDirectoryLoader(common.DATA_PATH).load())
    
    # Load TXT files
    for file in os.listdir(common.DATA_PATH):
        if file.endswith('.txt'):
            file_path = os.path.join(common.DATA_PATH, file)
            documents.extend(UnstructuredFileLoader(file_path).load())
            
    return documents

#function used to split docuuments into chunks
def split_documents(documents: list[Document]):

    #initialiase a new text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=common.CHUNK_SIZE,
        chunk_overlap=common.CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

    #split the document into chunks unsing the text splitter
    chunks = text_splitter.split_documents(documents)
    return chunks


#function to add the chunks into the chroma vector database
def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(persist_directory = common.CHROMA_PATH, embedding_function = common.embedding_function())

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    #call schutil function that deletes everything in the chroma path folder.
    if os.path.exists(common.CHROMA_PATH):
        shutil.rmtree(common.CHROMA_PATH)


if __name__ == "__main__":
    main()
