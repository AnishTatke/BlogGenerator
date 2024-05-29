import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import CTransformers, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

## Function to get response from models
def getLLamaResponse(model_type, input_text, num_words, blog_style):
    print(model_type, input_text, num_words, blog_style)
    if model_type == "llama":
        ## Llama model loaded locally in model\
        llm = CTransformers(
            model = "model\llama-2-7b-chat.ggmlv3.q8_0.bin",
            model_type = "llama",
            model_name = "llama",
            config = {
                'max_new_tokens': 256,
                'temperature': 0.01,
            }
        )
    elif model_type == "mixtral":
        llm = HuggingFaceEndpoint(
            repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1",
            temperature = 0.6,
            model_kwargs = {
                'max_length': 100,
            }
        )

    template = """
        Write a blog for {blog_style} community on the topic {input_text}
        with {num_words} words.
    """

    prompt = PromptTemplate(input_variables=['blog_style', 'input_text', 'num_words'], template=template)

    response = llm(prompt.format(blog_style=blog_style, input_text=input_text, num_words=num_words))
    return response


st.set_page_config(
    page_title="Blog Generator",
    page_icon="🔗",
    layout="centered",
    initial_sidebar_state="collapsed")


st.header("Blog Generator")

input_text = st.text_area("Enter your prompt here")

model_type = st.selectbox("Select Model", ["llama", "mixtral"], index = 0)

col1, col2 = st.columns([5, 5])
with col1:
    num_words = st.number_input("Number of words", min_value = 100, max_value = 1000, value = 500, step = 50)

with col2:
    blog_style = st.selectbox("Select Blog Style", ["Researchers", "Professionals", "Students"], index = 0)

submit = st.button("Generate Blog")
if submit:
    st.write(getLLamaResponse(model_type, input_text, num_words, blog_style))