import gradio as gr
from dotenv import load_dotenv

from answer import answer_question
from ingest import fetch_documents_from_upload, create_chunks, create_embeddings

load_dotenv(override=True)


def format_context(context):
    result = "<h2 style='color: #ff7800;'>Relevant Context</h2>\n\n"
    for doc in context:
        result += f"<span style='color: #ff7800;'>Source: {doc.metadata.get('source', 'unknown')}</span>\n\n"
        result += doc.page_content + "\n\n"
    return result

def chat(history):
    if not history:
        return history, "<i>No history yet. Ask a question after uploading documents.</i>"

    last_message = history[-1]["content"]
    prior = history[:-1]
    answer, context = answer_question(last_message, prior)
    history.append({"role": "assistant", "content": answer})
    return history, format_context(context)


def main():
    def put_message_in_chatbot(message, history):
        # Append user message to history as dict (role/content style)
        if not history:
            history = []
        history = history + [{"role": "user", "content": message}]
        return "", history

    def ingest_files(uploaded_files):
     
        docs = fetch_documents_from_upload(uploaded_files)
        if not docs:
            return "⚠️ No documents found in upload."

        chunks = create_chunks(docs)
        create_embeddings(chunks)

        return f"✅ Ingested {len(chunks)} chunks from {len(docs)} documents. You can now ask questions."


    with gr.Blocks(title="Expert Assistant") as ui:
        gr.Markdown("# 🏢 RAG Expert Assistant\nUpload documents, then ask questions based on them!")

        with gr.Row():
            with gr.Column(scale=1):
                # File upload area
                file_uploader = gr.File(
                    label="📁 Upload one or more documents",
                    file_count="multiple",
                    type="filepath",
                )
                ingest_status = gr.Markdown(
                    value="⬆️ Upload documents to build the knowledge base.",
                    label="Ingestion Status",
                )

            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="💬 Conversation",
                    height=600,
                )
                message = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything about the uploaded documents ...",
                    show_label=False,
                )

        # Context area under the row so it stretches nicely
        context_markdown = gr.Markdown(
            label="📚 Retrieved Context",
            value="*Retrieved context will appear here after you ask a question.*",
            height=300,
        )

        # Wire upload → ingest pipeline
        file_uploader.upload(
            ingest_files,
            inputs=file_uploader,
            outputs=ingest_status,
        )

        # Wire message box → add user message → call chat()
        message.submit(
            put_message_in_chatbot,
            inputs=[message, chatbot],
            outputs=[message, chatbot],
        ).then(
            chat,
            inputs=chatbot,
            outputs=[chatbot, context_markdown],
        )

    ui.launch(inbrowser=True)


if __name__ == "__main__":
    main()