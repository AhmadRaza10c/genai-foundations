from langchain_ollama import ChatOllama 
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

llm = ChatOllama(
    model="gemma3:1b",
    temperature=0,
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