import common
import re

from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

def main():
    chat_history = ChatHistory()
    print("""
          Commands:
          //exit - ends chat
          //hisory - prints full chat history
          //clear - clears chat history
          Chat started. """)
    
    while True:
        query_text = input("\nYou: ").strip()
        
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
            
        query_llm(query_text, chat_history)

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

def query_llm(query_text, chat_history):
    #create the prompt fo the LLM using the context text and query text.
    prompt = PromptTemplate.from_template(common.PROMPT_TEMPLATE).format(question=query_text, chat_history=chat_history.get_formatted_history())

    #call the LLM
    response_text = OllamaLLM(model=common.LLM).invoke(prompt)

    # Add the interaction to chat history
    chat_history.add_message("user", query_text)
    chat_history.add_message("MATE", response_text)

    if "./search" in response_text:
        search_key = re.findall('"([^"]*)"', response_text)[0]
        searchresult_text = rag_search(search_key)
        chat_history.add_message("SEARCH RESULT", searchresult_text)
        prompt_after_search = PromptTemplate.from_template(common.PROMPT_TEMPLATE_AFTER_SEARCH).format(chat_history=chat_history.get_formatted_history())
        response_text = OllamaLLM(model=common.LLM).invoke(prompt_after_search)
        chat_history.add_message("MATE", response_text)

    print(f"\nMATE: {response_text}")


def rag_search(search_key):
    print("\nsearching lecture database...")
    #Search the DB, and return the 5 most similar chunks of context.
    results = Chroma(persist_directory= common.CHROMA_PATH, embedding_function= common.embedding_function() ).similarity_search_with_score(search_key, k=common.NUM_CHUNKS)
    #create the search result text for the LLM. It will be the chunks of text, seperated by new lines and three dashes.
    searchresult_text = "\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'Unknown')})\n{doc.page_content}" for doc, _score in results])
    print("\n".join([f"Searched sources: {doc.metadata.get('source','')}" for doc, _score in results]))
    return searchresult_text

if __name__ == "__main__":
    main()