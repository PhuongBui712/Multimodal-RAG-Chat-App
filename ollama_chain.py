from langchain_community.llms import Ollama
from langchain.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain, create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import Document

from pinecone import Pinecone, ServerlessSpec, PodSpec
from langchain_pinecone import PineconeVectorStore
from langchain_chroma import Chroma
from pdf_handler import extract_pdf, load_pdf, split_pdfs

from utils import load_config
from dotenv import load_dotenv
import os


def format_docs(docs: list[Document]):
    return '\n\n'.join(doc.page_content for doc in docs)


def setup_pinecone(index_name, embedding_model, embedding_dim, metric='cosine', use_serverless=True):
    load_dotenv()
    pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
    if use_serverless:
        spec = ServerlessSpec(cloud='aws', region='us-east-1')
    else:
        spec = PodSpec()

    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name)

    pc.create_index(
        index_name,
        dimension=embedding_dim,
        metric=metric,
        spec=spec
    )

    db = PineconeVectorStore.from_documents(index_name=index_name, embedding=embedding_model)
    return db


def setup_chroma(embedding_model, directory=None):
    if not directory:
        directory = './.cache/database/'
    os.makedirs(directory, exist_ok=True)

    db = Chroma(embedding_function=embedding_model, persist_directory=directory)
    return db


def load_vector_database(database: str, persist_directory=None):
    # initialize embedding
    embedding_model = OllamaEmbeddings(model='nomic-embed-text:latest')

    if database == 'pinecone':
        # initialize pinecone
        index_name = 'rag-pdf'
        db = setup_pinecone(index_name, embedding_model, 'cosine')

    else:
        db = setup_chroma(embedding_model, persist_directory)

    return db


class OllamaChain:
    def __init__(self, config_path, chat_memory) -> None:
        prompt = PromptTemplate(
            template="""<|begin_of_text|>
            <|start_header_id|>system<|end_header_id|>
            You are a honest and unbiased AI assistant
            <|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            Previous conversation={chat_history}
            Question: {input} 
            Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=['chat_history', 'input']
        )

        self.memory = ConversationBufferWindowMemory(
            memory_key='chat_history',
            chat_memory=chat_memory,
            k=3,
            return_messages=True
        )

        config = load_config()
        llm = Ollama(**config['ollama_model'])
        # llm = Ollama(model='llama3:latest', temperature=0.75, num_gpu=1)

        self.llm_chain = LLMChain(prompt=prompt, llm=llm, memory=self.memory, output_parser=StrOutputParser())
        # runnable = prompt | llm

    def run(self, user_input):
        response = self.llm_chain.invoke(user_input)

        return response['text']


class OllamaRAGChain:
    def __init__(self, config_path, chat_memory, uploaded_pdfs=None):
        # initialize vector db
        self.vector_db = load_vector_database('chroma')
        if uploaded_pdfs:
            self._update_knowledge_base(uploaded_pdfs)

        # initialize llm
        config = load_config()
        self.llm = Ollama(**config['ollama_model'])

        # initialize memory
        self.chat_memory = chat_memory

        # initialize sub chain with history message
        contextual_q_system_prompt = """Given a chat history and the latest user question which might reference context in the\
        chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer \
        the question, just reformulate it if needed and otherwise return it as is."""

        self.contextual_q_prompt = ChatPromptTemplate.from_messages(
            [
                ('system', contextual_q_system_prompt),
                MessagesPlaceholder('chat_history'),
                ('human', '{input}'),
            ]
        )

        self.history_aware_retriever = create_history_aware_retriever(
            self.llm, self.vector_db.as_retriever(), self.contextual_q_prompt
        )

        # initialize qa chain
        qa_system_prompt = """You are an assistant for question-answering tasks. Use the following pieces of retrieved\
        context to answer the question. If you don't know the answer, just say that you don't know. \
        Use three sentences maximum and keep the answer concise.\
        
        {context}"""
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ('system', qa_system_prompt),
                MessagesPlaceholder('chat_history'),
                ('human', '{input}'),
            ]
        )

        self.question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        rag_chain = create_retrieval_chain(self.history_aware_retriever, self.question_answer_chain)

        self.conversation_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            lambda session_id: chat_memory,
            input_messages_key='input',
            history_messages_key='chat_history',
            output_messages_key='answer'
        )

    def run(self, user_input):
        config = {"configurable": {"session_id": "any"}}
        response = self.conversation_rag_chain.invoke({'input': user_input}, config)
        print(response['context'])

        return response['answer']

    def update_chain(self, uploaded_pdf):
        self._update_knowledge_base(uploaded_pdf)
        self.history_aware_retriever = create_history_aware_retriever(
            self.llm, self.vector_db.as_retriever(), self.contextual_q_prompt
        )
        self.conversation_rag_chain = RunnableWithMessageHistory(
            create_retrieval_chain(self.history_aware_retriever, self.question_answer_chain),
            lambda session_id: self.chat_memory,
            input_messages_key='input',
            history_messages_key='chat_history',
            output_messages_key='answer'
        )

    def _update_knowledge_base(self, uploaded_pdf):
        files = extract_pdf(uploaded_pdf)
        for f in files:
            # add to database
            doc = load_pdf(f)
            chunks = split_pdfs(doc)
            self.vector_db.add_documents(chunks)

            # remove file
            os.remove(f)