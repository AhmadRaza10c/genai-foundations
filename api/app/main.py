import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

load_dotenv()

llm = init_chat_model(
    model=os.getenv("MODEL_NAME"),
    temperature=0.1,
)

messages = [
    SystemMessage(content="You are a helpful AI assistant."),
]

while True:
    Question = input("Prompt: ").strip()
    
    if Question in ["exit", "quit"]:
        break
        
    if not Question.strip():
        continue

    messages.append(HumanMessage(content=Question))
    response = llm.invoke(messages)
    messages.append(AIMessage(content=response.content))
        
    print(f"Bot: {response.content}")