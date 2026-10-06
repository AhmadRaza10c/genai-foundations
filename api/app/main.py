import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
# from langchain_ollama import Ollama
# from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

load_dotenv()

llm = init_chat_model(
    model=os.getenv("MODEL_NAME"),
    temperature=1.0,
)

history = [{"role": "system", "content": "You are a helpful AI assistant."}]

# messages = [
#     SystemMessage(content="You are a helpful AI assistant."),
# ]

print("Welcome to cue chat!")
while True:
    Question = input("Prompt: ").strip()
    
    if Question in ["exit", "quit"]:
        break
        
    if not Question.strip():
        continue

    history.append({
        "role": "user",
          "content": Question
    })
    response = llm.invoke(history)
    history.append(response.content)

    print(f"Bot: {response.content}")