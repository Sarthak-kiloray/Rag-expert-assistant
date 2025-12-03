from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document
from dotenv import load_dotenv
from collections.abc import Mapping, Sequence
from typing import List, Any

load_dotenv(override=True)

MODEL = "gpt-4.1-nano"
DB_NAME = str(Path(__file__).parent.parent / "vector_db")

# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
RETRIEVAL_K = 10

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing a smart agent that can answer questions about the uploaded documents.
You are chatting with a user about the uploaded documents.
Go through all the chunks in depth before answering the question. Try to use the context before self training.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""
RETRIEVAL_K = 4
# vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
# retriever = vectorstore.as_retriever()
llm = ChatOpenAI(temperature=0, model_name=MODEL)

def get_retriever():
    """
    Build a fresh retriever from the current Chroma collection on disk.
    This avoids stale collection IDs after re-ingesting.
    """
    vectorstore = Chroma(
        persist_directory=DB_NAME,
        embedding_function=embeddings,
    )
    return vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})

def fetch_context(question: str) -> list[Document]:
    """
    Retrieve relevant context documents for a question.
    """
    if isinstance(question, list):
        question = "\n".join(map(str, question))
    elif not isinstance(question, str):
        question = str(question)

    retriever = get_retriever()
    return retriever.invoke(question)


def combined_question(question: str, history) -> str:
    """
    Build a retrieval query from the chat history + current question.

    Gradio Chatbot history usually looks like:
        [
            ["user msg 1", "assistant reply 1"],
            ["user msg 2", "assistant reply 2"],
            ...
        ]
    We just stitch together previous user messages plus the new question.
    """
    if isinstance(question, list):
        question_str = "\n".join(map(str, question))
    else:
        question_str = str(question)

    if not history:
        return question_str

    prior_user_msgs: List[str] = []

    if isinstance(history, Sequence) and not isinstance(history, (str, bytes)):
        for turn in history:
            # Case A: dict with role/content
            if isinstance(turn, Mapping):
                role = turn.get("role")
                content = turn.get("content")
                if role == "user" and content:
                    prior_user_msgs.append(str(content))

            # Case B: simple [user, assistant] pair
            elif isinstance(turn, Sequence) and not isinstance(turn, (str, bytes)):
                if len(turn) > 0 and turn[0]:
                    prior_user_msgs.append(str(turn[0]))

    # Fallback if history is some other type
    if not prior_user_msgs and not isinstance(history, (str, bytes)):
  
        history_str = str(history).strip()
        if history_str:
            prior_user_msgs.append(history_str)

    prior_text = "\n".join(prior_user_msgs).strip()

    if not prior_text:
        return question_str

    return prior_text + "\n" + question_str


def answer_question(question: str, history=None) -> tuple[str, list[Document]]:
    """
    Answer the given question with RAG; return the answer and the context documents.
    `history` is the chat history coming from the Gradio Chatbot component.
    """
    if history is None:
        history = []

    combined = combined_question(question, history)
    docs = fetch_context(combined)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content, docs