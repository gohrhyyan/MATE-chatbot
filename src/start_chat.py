import common
import re

from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

class ChatMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class ChatHistory:
    def __init__(self):
        self.messages = []
    
    # adds a message to the chat history
    def add_message(self, role, content):
        self.messages.append(ChatMessage(role, content))
        if role == "MATE":
           print(f"\nMATE: {content}")
        #if role != "USER":
        #    print(f"\n{role}: {content}")

        # Keep only the last CHAT_CONTEXT_LENGTH messages
        if len(self.messages) > common.CHAT_CONTEXT_LENGTH:
            self.messages = self.messages[-common.CHAT_CONTEXT_LENGTH:]
    
    # retrieve the chat history for the prompt
    def get_formatted_history(self):
        formatted = []
        for msg in self.messages:
            formatted.append(f"***{msg.role}***: {msg.content}")
        return "\n".join(formatted)

    #clears the chat history    
    def clear(self):
        self.messages = []

chat_history = ChatHistory()

def main():
    print("""
          Commands:
          //exit - ends chat
          //hisory - prints full chat history
          //clear - clears chat history
          Chat started. """) 
    while True:
        query_text = input("\nUser: ").strip()

        if query_text.lower() == "//exit":
            print("\nGoodbye!")
            break
        elif query_text.lower() == "//clear":
            chat_history.clear()
            print("\nChat history cleared.")
            continue
        elif query_text.lower() == "//history":
            print(f"\n{chat_history.get_formatted_history()}")
            continue
        elif not query_text:
            continue
            
        process_input(query_text, chat_history)


def process_input(query_text, chat_history):
    chat_history.add_message("USER", query_text)
    llm = OllamaLLM(model=common.LLM)

    # Initial prompt setup
    current_chat_history = chat_history.get_formatted_history()

    response = generate_response(
        query_text, 
        current_chat_history, 
        llm
    )
    
    # Add the final response to chat history
    chat_history.add_message("MATE", response)

def generate_response(query_text, current_chat_history, llm):
    """
    Generates responses, handling both search and reasoning steps.
    Args:
        query_text: The user's query
        current_chat_history: Formatted chat history
        llm: The language model instance
    Returns:
        str: The final response from MATE
    """
    prompt = PromptTemplate.from_template(common.PROMPT_TEMPLATE).format(
        question=query_text,
        chat_history=current_chat_history,
        )
    response = llm.invoke(prompt)

    if "./search" in response:
        chain_of_thought = ChatHistory()
        while "./search" in response:
            # Record the thought that led to the search
            chain_of_thought.add_message("MATE'S THOUGHTS", response)

            # Extract and perform search
            search_match = re.search(r'\.\/search\s*"([^"]*)"', response)
            if search_match:
                search_key = search_match.group(1)
                # Indicate in chat history that a search was performed
                chat_history.add_message("MATE", f"Searching for: {search_key}")
                search_results = rag_search(search_key)

                # Add search results to chain of thought 
                chain_of_thought.add_message("SEARCH RESULTS", search_results)

            # Format prompt and get LLM response
            response = llm.invoke(PromptTemplate.from_template(common.PROMPT_TEMPLATE_AFTER_SEARCH).format(
                question=query_text,
                chat_history=current_chat_history,
                chain_of_thought=chain_of_thought.get_formatted_history()
            ))

        chain_of_thought.add_message("MATE'S THOUGHTS", response)
        prompt = PromptTemplate.from_template(common.PROMPT_TEMPLATE_AFTER_REASONING).format(
            question=query_text,
            chat_history=current_chat_history,
            chain_of_thought=chain_of_thought.get_formatted_history()
        )
        response = llm.invoke(prompt)
        chain_of_thought.clear()
    #print(prompt)
    return response


def rag_search(search_key):
    print("\nsearching lecture database...")
    #Search the DB, and return the most similar chunks of context.
    results = Chroma(persist_directory= common.CHROMA_PATH, embedding_function= common.embedding_function() ).similarity_search_with_score(search_key, k=common.NUM_CHUNKS)
    
    #create the search result text for the LLM. It will be the chunks of text, seperated by new lines and three dashes.
    searchresult_text = "\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'Unknown')})\n{doc.page_content}" for doc, _score in results])
    print("\n".join([f"Searched sources: {doc.metadata.get('source','')} (Page {doc.metadata.get('page', 'Unknown')})" for doc, _score in results]))
    return searchresult_text

if __name__ == "__main__":
    main()
