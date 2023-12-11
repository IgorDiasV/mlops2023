import gradio as gr
from classification_text import Classifytext

classification = Classifytext()

tela = gr.Interface(fn=classification.predict_input_user,
                    inputs=gr.Textbox(lines=2,
                                      placeholder="Digite o texto...",
                                      label="Texto"),
                    outputs=gr.Textbox(lines=1,
                                       label="Tipo de Texto"))

if __name__ == "__main__":
    tela.launch(show_api=False)