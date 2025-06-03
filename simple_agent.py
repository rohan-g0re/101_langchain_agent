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

#load environment variables
load_dotenv()




"""
--------------------------------STEP 1 --------------------------------
Initialize the LLM, which acts as the "brain" of the agent

"""

llm = ChatGoogleGenerativeAI(model = "gemini-2.0-flash", api_key = os.getenv("GOOGLE_API_KEY"))

print ("LLM initialized")
print()




"""
--------------------------------STEP 2 --------------------------------

The agent would be using different TOOLS, to complete the tasks given to it.
Therefore we will define the tools here. 

# A tool needs:
    1. a name, 
    2. a function (what it does), 
    3. a description (CRUCIAL for the LLM).

# The LLM uses the description to decide WHEN to use the tool and WHAT to feed into it.
"""

search_tool = DuckDuckGoSearchRun()

tools = [
    Tool(
        name = "Web Search",
        func = search_tool.run,
        description = "Useful for when you need to answer questions about current events,\
            facts, or any information not in your knowledge base.\
            Input should be a search query."
    )
]

"""
description is IMPORTANT because:

The LLM reads this description to understand what the tool does and when it should be used.
A good description is key to a well-functioning agent.

"""


"""
-------------------------------- STEP 3: Creating the agent --------------------------------

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
-------------------------------- STEP 4: Creating the AgentExecutor --------------------------------

this actually runs the agent.

It takes the agent and tools, and handles the "think-act-observe" loop.

we will keep verbose=True will print out the agent's thoughts and actions, which is great for understanding!

"""

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print ("AgentExecutor created")




"""
-------------------------------- STEP 5: MAIN: for running the agent --------------------------------

"""


if __name__ == "__main__":
    print("Running the agent...")

    question = "What is the latest Sidemen video on youtube?"

    try:
        response = agent_executor.invoke({"input": question})
        print("Response:")
        print(response["output"])
    except Exception as e:
        print(f"An error occurred: {e}")



