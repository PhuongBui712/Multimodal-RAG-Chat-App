import streamlit as st

from ollama_chain import OllamaChain, OllamaRAGChain
from llama_cpp_chains import LlamaChain
from pdf_handler import extract_pdf
from langchain.memory import StreamlitChatMessageHistory


def clear_input_field():
    # store the question
    st.session_state.user_question = st.session_state.user_input
    # clear the variable
    st.session_state.user_input = ""


def set_send_input():
    st.session_state.send_input = True
    clear_input_field()


@st.cache_resource
def load_llm_chain(config_path, _chat_memory):
    return LlamaChain(config_path, _chat_memory)


@st.cache_resource
def load_ollama_chain(config_path, _chat_memory):
    return OllamaChain(config_path, _chat_memory)


@st.cache_resource
def load_ollama_rag_chain(config_path, _chat_memory, uploaded_pdfs=None):
    return OllamaRAGChain(config_path, _chat_memory, uploaded_pdfs)


def clear_cache():
    st.cache_resource.clear()


def initial_session_state():
    st.session_state.send_input = False
    st.session_state.knowledge_file = 0


def main():
    # Initialize
    # Title
    st.title('Local Chat App')
    chat_container = st.container()

    # sidebar
    # st.sidebar.title('Chat Session')

    # file upload
    uploaded_pdf = st.sidebar.file_uploader('Upload your pdf files',
                                            type='pdf',
                                            accept_multiple_files=True,
                                            key='upload_file')

    # Input objects
    user_input = st.text_input('Type your message here', key='user_input', on_change=set_send_input)
    send_button = st.button('Send', key='send_button')

    # Session state
    if 'send_input' not in st.session_state:
        initial_session_state()
    # ----------------------------------------------------------------------------------------------------

    chat_history = StreamlitChatMessageHistory(key='history')

    with chat_container:
        for msg in chat_history.messages:
            st.chat_message(msg.type).write(msg.content)

    # llm_chain = None
    if uploaded_pdf:
        # llm_chain = load_ollama_rag_chain('config.yaml', chat_history)
        llm_chain = load_ollama_rag_chain('config.yaml', chat_history, uploaded_pdfs=uploaded_pdf)
    else:
        llm_chain = load_ollama_chain('config.yaml', chat_history)

    # we use "or" operation here because user can press 'Enter' instead of 'Send' button
    if send_button or st.session_state.send_input:
        if st.session_state.user_question != "":
            with chat_container:
                st.chat_message('user').write(st.session_state.user_question)
                llm_response = llm_chain.run(user_input=st.session_state.user_question)
                st.chat_message('ai').write(llm_response)


if __name__ == '__main__':
    main()
