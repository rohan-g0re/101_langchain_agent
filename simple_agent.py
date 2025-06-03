import os 
os.environ["LANGCHAIN_TRACING_V2"] = "false"
from dotenv import load_dotenv


#lanchain specific imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun

from langchain import hub # for loading agent and potentially prompt templates

#python specific imports
from pydantic import BaseModel, Field

# --------------------

# import for vector db integration

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA   # a chain for retrieval which uses a vector database






#load environment variables
load_dotenv()




"""
--------------------------------STEP 1 --------------------------------
1. Initialize the LLM, which acts as the "brain" of the agent
2. Initializing embeddings model --> google 

"""

llm = ChatGoogleGenerativeAI(model = "gemini-2.0-flash", api_key = os.getenv("GOOGLE_API_KEY"))

print ("LLM initialized")

embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

print ("Embeddings model initialized")



"""
----------STEP/PHASE 2 --> loading knowledge base and setting up vector database ---------

"""

# 2.1 loading text files from knowledge_base directory

loader = DirectoryLoader("./knowledge_base/", glob = "*.txt", loader_cls = TextLoader, show_progress = True)

documents = loader.load()

print(f"Loaded {len(documents)} documents.")



# 2.2 splitting the documents into chunks

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)

docs = text_splitter.split_documents(documents)

print (F"Split into {len(docs)} chunks.")



# 2.3 downloading and setting uup the embedding model - FAISS

try:
    vector_store = FAISS.from_documents(docs, embeddings)
    print ("Vector store created successfully")
except Exception as e:
    print (f"An error occurred while creating the vector store: {e}")


# 2.4 setting up the retriever from the vector db 

retriever = vector_store.as_retriever(search_kwargs = {"k": 3}) # k is the number of --TOP RELEVANT-- chunks to retrieve

print ("Retriever created successfully")











"""
--------------------------------STEP 3 --------------------------------

The agent would be using different TOOLS, to complete the tasks given to it.
Therefore we will define the tools here. 

# A tool needs:
    1. a name, 
    2. a function (what it does), 
    3. a description (CRUCIAL for the LLM).

# The LLM uses the description to decide WHEN to use the tool and WHAT to feed into it.
"""

search_tool = DuckDuckGoSearchRun()

knowledge_base_tool = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type = "stuff",
    retriever = retriever
)

tools = [
    Tool(
        name = "Web Search",
        func = search_tool.run,
        description = "Useful for when you need to answer questions about social media platforms like tiktok, instagram and youtube\
            or any information not realted to finance and trading.\
            Input should be a search query."
    ),
    Tool(
        name = "Knowledge Base",
        func = knowledge_base_tool.invoke,
        description =  "Use this tool when you need to answer questions about financial strategies, trading strategies, and latest financial developments. Input should be a full question." 
    )
]

# description is IMPORTANT because:

""" The LLM reads this description to understand what the tool does and when it should be used.
A good description is key to a well-functioning agent."""


print("Tools created.")



"""
-------------------------------- STEP 4: Creating the agent --------------------------------

1. we need a template, which tellls the agent ho to behave -- >
    APPARENTLY, this is called a "prompt template"

2. langchain HUB has pre-defined templates --> we use "hwchase17/react" is a common one for ReAct agents.

"""

prompt_template = hub.pull("hwchase17/react")

print("Prompt template AHEAD:")
print()
#log the prompt template
print (prompt_template)
print()

# This function creates the agent itself by combining:
# - The LLM (our Gemini model)
# - The available tools
# - The prompt template

agent = create_react_agent(llm, tools, prompt_template)

print ("Agent created")



"""
-------------------------------- STEP 5: Creating the AgentExecutor --------------------------------

this actually runs the agent.

It takes the agent and tools, and handles the "think-act-observe" loop.

we will keep verbose=True will print out the agent's thoughts and actions, which is great for understanding!

"""

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True # Helpful for debugging LLM output parsing
)


print ("AgentExecutor created")




"""
-------------------------------- STEP 6: MAIN: for running the agent --------------------------------

"""


if __name__ == "__main__":
    print("Running the agent...")

    # question = "What is the latest Sidemen video on youtube?"

    questions = [
        "What is a key challenge related to data in ESG investing?",
        "How is Natural Language Processing (NLP) utilized in modern trading sentiment analysis?",
        "What is 'model decay' in the context of algorithmic trading?",
        
        "What are YouTube's current monetization eligibility requirements for creators?",
        "How does YouTube's algorithm determine video recommendations for users?",
        "Who is the most subscribed individual creator on YouTube today?"
    ]

    for question in questions:
        print(f"Question: {question}")
        print("-"*100)

        try:
            response = agent_executor.invoke({"input": question}) # the input MUST be a dictionary
            print("Response:")
            print(response["output"])
            print("-"*100)
        except Exception as e:
            print(f"An error occurred: {e}")
            print("-"*100)



