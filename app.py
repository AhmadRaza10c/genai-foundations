import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# =============================================================================
# AI LOGIC LAYER
# =============================================================================

class AIChatBot:
    """
    Encapsulates the AI logic to separate it from the Streamlit UI.
    Keeps the logic unchanged from main.py.
    """
    def __init__(self, model_name="gemma3:1b", temperature=0):
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
        )

    def get_response(self, messages):
        """
        Returns a generator for streaming the response content.
        """
        try:
            # Using .stream() for modern UX as per plan
            for chunk in self.llm.stream(messages):
                yield chunk.content
        except Exception as e:
            yield f"Error: {str(e)}"

# =============================================================================
# UI LAYER (Streamlit)
# =============================================================================

def main():
    # Page configuration for a professional look
    st.set_page_config(
        page_title="AI Assistant",
        page_icon="🤖",
        layout="centered"
    )

    st.title("🤖 AI Assistant")
    st.markdown("---")

    # Initialize AI Bot
    # Using @st.cache_resource to avoid re-initializing the LLM on every rerun
    @st.cache_resource
    def load_bot():
        return AIChatBot()

    bot = load_bot()

    # Initialize Chat History in session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful AI assistant."),
        ]

    # Display chat messages from history
    # We skip the SystemMessage in the UI for a cleaner experience
    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        elif isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)

    # User input
    if prompt := st.chat_input("How can I help you today?"):
        # Add user message to chat history
        st.session_state.messages.append(HumanMessage(content=prompt))

        # Display user message immediately
        st.chat_message("user").write(prompt)

        # Generate and display AI response with streaming
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            # Streaming state
            with st.spinner("Thinking..."):
                try:
                    for chunk in bot.get_response(st.session_state.messages):
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")

                    response_placeholder.markdown(full_response)
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

            # Save AI response to history
            st.session_state.messages.append(AIMessage(content=full_response))

if __name__ == "__main__":
    main()
