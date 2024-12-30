from langchain_ollama import OllamaEmbeddings
import os

def get_db_paths():
    """Returns a dictionary of database paths for each top-level category"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data")
    chroma_base = os.path.join(base_dir, "chroma")
    
    # Create dictionary of database paths
    db_paths = {}
    
    # Walk through first two levels of directories
    for root, dirs, files in os.walk(data_path):
        depth = root[len(data_path):].count(os.sep)
        if depth <= 1:  # Only process top-level and immediate subdirectories
            relative_path = os.path.relpath(root, data_path)
            if depth == 0:
                continue  # Skip the root data directory
            
            # Get the top-level directory name
            top_dir = relative_path.split(os.sep)[0]
            
            # Create path for this category's database
            db_path = os.path.join(chroma_base, top_dir)
            db_paths[top_dir] = db_path
            
    return db_paths

def embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#constants
CHROMA_PATH = os.path.join(BASE_DIR, "chroma")
DATA_PATH = os.path.join(BASE_DIR, "data")
LLM = "llama3.1:8b"

#tuning
CHAT_CONTEXT_LENGTH = 100
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
NUM_CHUNKS = 2 #number of context chunks to provide to the LLM

PROMPT_TEMPLATE = """
CURRENT CHAT:
{chat_history}

SYSTEM PROMPT:
You are MATE, short for MATErials. 

This is the user's query:
"{question}"

Determine whether this query explicitly requests information about materials science and engineering, and how to respond to the user.
Respond with your thoughts between <MATEthoughts> tags like this:
<MATEthoughts>
[Your thoughts here]
</MATEthoughts>

Then, ONLY after </MATEthoughts> follow these RULES:
1. For general conversation:
 - Respond to the user's query, following the conversation flow in CURRENT CHAT.

2. If the the user's query specifically requests information about materials science and engineering:
 - Respond with search command only: //search "keywords"
 - Do not respond with anything other than the command and keywords, no other words.
 - Never use "./search" by itself - always include keywords in quotes.
 - Do not include "Materials", "Science" or "Engineering" in your search keywords.
 - Keywords must be an concise summary of the user's query in 5-words or less.

DO NOT include "***MATE***" in your response. It is automatically prepended.
"""

PROMPT_TEMPLATE_AFTER_SEARCH = """
CURRENT CHAT:
{chat_history}

{chain_of_thought}

SYSTEM PROMPT:
You are MATE (MATErials).
Your goal is to plan a response to ***USER***'s query in "CURRENT CHAT": "{question}" in the context of {context}.
You've just completed a search with results in ***SEARCH RESULTS***.

STOP SEARCHING IF:
   - Core principles and applications are covered
   - You catch yourself in an infinite loop of off-topic or redundant search results

FOR ADDITIONAL SEARCH ONLY IF NECESSARY:
   - Target missing fundamental concepts
   - Avoid repeating previous search terms
   - Format: //search "keywords"

You are now thinking as ***MATE'S THOUGHTS*** (not shown to user), refer to yourself as "MATE", refer to the user as "the user"

DO NOT Generate the actual response to the user, instead DO:
1. Plan steps for a response that promotes a deep understanding from first principles, focused on the user's query.
2. Consider if one more search is needed, if so, excecute the search command: //seacrch "new keywords here"
***MATE'S THOUGHTS***:
"""

PROMPT_TEMPLATE_AFTER_REASONING = """
CURRENT CHAT:
{chat_history}

{chain_of_thought}

SYSTEM PROMPT:
You are MATE (MATErials). Structure your response following these principles:
   - Start with basic definitions
   - Progress logically from first principles understanding.
   - Include relevant equations and derivations
   - Use clear section headings and bullet points.

DO NOT include your further ***MATE'S THOUGHTS*** or your other internal thinking in your response.
Respond as ***MATE*** using the reasoning in ***MATE'S THOUGHTS*** to address ***USER***'s query in "CURRENT CHAT": "{question}" in the context of {context}.
***MATE***:
"""