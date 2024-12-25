import common

from typing import List, Tuple
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

def main():
    chat_history = ChatHistory()
    print("Chat started. Type './exit' to end or './clear' to clear history.")
    
    while True:
        query_text = input("\nYou: ").strip()
        
        if query_text.lower() == "./exit":
            print("Goodbye!")
            break
        elif query_text.lower() == "./clear":
            chat_history.clear()
            print("Chat history cleared.")
            continue
        elif not query_text:
            continue
            
        query_rag(query_text, chat_history)

class ChatMessage:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

class ChatHistory:
    def __init__(self):
        self.messages: List[ChatMessage] = []
    
    # adds a message to the chat history
    def add_message(self, role: str, content: str):
        self.messages.append(ChatMessage(role, content))

        # Keep only the last CHAT_CONTEXT_LENGTH messages
        if len(self.messages) > common.CHAT_CONTEXT_LENGTH:
            self.messages = self.messages[-common.CHAT_CONTEXT_LENGTH:]
    
    # retrieve the chat history for the prompt
    def get_formatted_history(self) -> str:
        formatted = []
        for msg in self.messages:
            formatted.append(f"{msg.role}: {msg.content}")
        return "\n".join(formatted)

    #clears the chat history    
    def clear(self):
        self.messages = []

def query_rag(query_text: str, chat_history: ChatHistory):
    #Search the DB, and return the 5 most similar chunks of context.
    results = Chroma(persist_directory= common.CHROMA_PATH, embedding_function= common.embedding_function() ).similarity_search_with_score(query_text, k=5)

    #create the context text for the LLM. It will be the chunks of text, seperated by new lines and three dashes.
    context_text = "\n\n---\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'Unknown')})\n{doc.page_content}" for doc, _score in results])

    #create the prompt fo the LLM using the context text and query text.
    prompt = PromptTemplate.from_template(common.PROMPT_TEMPLATE).format(context=context_text, question=query_text, chat_history=chat_history.get_formatted_history())
    print(prompt)
    #call the LLM
    response_text = OllamaLLM(model="llama3.2:1b").invoke(prompt)

    # Add the interaction to chat history
    chat_history.add_message("user", query_text)
    chat_history.add_message("AI", response_text)

    print(f"Response: {response_text}")

if __name__ == "__main__":
    main()
