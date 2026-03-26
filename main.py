from dotenv import load_dotenv
from importlib.metadata import version
load_dotenv()


core_version = version("langchain-core")
lg_version = version("langgraph")
from langchain_openai import ChatOpenAI


print (f"langchain-core version: {core_version}")
print (f"langchain version: {lg_version}")

def main():
    print("Hello from langchain-langgraph!")

    # Test OpenAI
    llm = ChatOpenAI(model_name = "gpt-4o-mini", temperature = 0)
    response = llm.invoke ("Say 'setup complete' in one word")
    print (f"Response from ChatOpenAI: {response}")

    print ("Setup Complete")

if __name__ == "__main__":
    main()
