import sys
from search import search_prompt

def main():
    """
    Main function for the CLI chat loop.
    """
    # Attempt to initialize the RAG chain
    chain = search_prompt()

    if not chain:
        print("Could not start the chat. Check the initialization errors above.")
        return
    
    print("\n" + "="*80)
    print("MBA IA FullCycle - Semantic Ingestion and Search with LangChain and Postgres")
    print("Rules: Answers ONLY based on the PDF provided during ingestion.")
    print("Type 'sair' or 'exit' to quit.")
    print("="*80 + "\n")

    # Main chat loop
    while True:
        try:
            user_input = input("QUESTION: ").strip()
            
            # Exit condition
            if user_input.lower() in ("sair", "exit"):
                print("Closing the chat. Goodbye!")
                break
            
            if not user_input:
                continue

            # Invocation of the LangChain Runnable Chain
            # We pass the user's question to the chain
            print("SEARCHING...")
            response = chain.invoke(user_input)
            
            # Display the formatted response
            print(f"\nANSWER: {response}\n")

        except KeyboardInterrupt:
            # Capture Ctrl+C
            print("\nClosing the chat. Goodbye!")
            break
        except Exception as e:
            # Generic error handling during execution
            print(f"\nAn unexpected error occurred: {e}")
            break

if __name__ == "__main__":
    # Ensures the script is running in the terminal context.
    if sys.version_info < (3, 12):
        print("This script requires Python 3.12 or higher.")
    else:
        main()