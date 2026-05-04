# ~/course_rag/app.py
import gradio as gr
import yaml
from pathlib import Path
from rag_pipeline import ask, load_config

config = load_config()
subjects = ["Auto-detect"] + list(config.keys())

def respond(question, subject_choice, history):
    force = None if subject_choice == "Auto-detect" else subject_choice
    return ask(question, config, force_subject=force)

with gr.Blocks(title="Course Assistant") as demo:
    gr.Markdown("## CS Course Assistant")
    gr.Markdown("Ask anything from your course materials.")

    with gr.Row():
        subject_dd = gr.Dropdown(
            choices=subjects,
            value="Auto-detect",
            label="Subject",
            scale=1,
        )
        chatbot = gr.Chatbot(scale=4)

    msg = gr.Textbox(placeholder="Ask a question...", label="Your question")
    msg.submit(respond, [msg, subject_dd, chatbot], chatbot)

demo.launch(server_name="0.0.0.0", server_port=7860)










