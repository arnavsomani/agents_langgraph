from dotenv import load_dotenv
load_dotenv()

from langchain_core import __version__ as core_version
from langgraph import __version__ as lg_version
from langchain_openai import ChatOpenAI

core_version = version("langchian-core")
lg_version = version("langgraph")

def main():
    print("Hello from langchain-langgraph!")


if __name__ == "__main__":
    main()
