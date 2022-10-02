import asyncio
import gc
import logging
import os
import pandas as pd
import psutil
import streamlit as st
from PIL import Image
from streamlit import components
from streamlit.caching import clear_cache
from bs4 import BeautifulSoup
from haystack.pipeline import  ExtractiveQAPipeline, Pipeline
from haystack.document_store.memory import InMemoryDocumentStore
from haystack.retriever.sparse import TfidfRetriever
from haystack.reader import FARMReader


os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)

def print_memory_usage():
    logging.info(f"RAM memory % used: {psutil.virtual_memory()[2]}")

def write_sidebar_footer():
    st.sidebar.markdown('''
        ### About Me
        - AI Enthusisast
        - Foodie
        - Lone monk
        - Traveller

        ## Know more
        - Follow: https://abhi.ml
    ''')


@st.cache(allow_output_mutation=True, suppress_st_warning=True, max_entries=1)
def load_model(model_name):
    return None
    # return (
    #     AutoModelForSequenceClassification.from_pretrained(model_name),
    #     AutoTokenizer.from_pretrained(model_name),
    #     pipeline("zero-shot-classification"),
    #     spacy.load('en_core_web_lg'),

    # )


def write_ui():


    ############### add widgets in sidebar ##################
    # add a course5 logo
    st.sidebar.title("Know me")
    img = Image.open(os.path.join('abhi.jpg'))
    st.sidebar.image(img)
    # sel = st.sidebar.selectbox('Select', ["Named Entity Recognition","Token Attributes Extraction","Sentiment Analysis","Classification"])
    # uncomment the options below to test out the app with a variety of classification models.
    write_sidebar_footer()
    name_email_cols = st.beta_columns(2)
    name = name_email_cols[0].text_input(
        "Please enter your name"
    )

    email = name_email_cols[1].text_input(
        "Please enter your email"
    )

    if len(email) >0:
        st.write("Thanks a lot for entering your details. Now you can ask any questions you have. Write your question below. HAVE FUN !!!!!")
        text = st.text_area(
            "Enter text to be interpreted",
            "Where do you work?",
            height=100,
        )
        res = search_pipeline(text)
        # st.markdown(res[0])

        st.header(res["answers"][0]["answer"])
        st.write(res["answers"][0]["context"])
        st.markdown("        ")
        if st.button('Not satisfied with the answer. Press here'):
            feedback = "no"
        else:
            feedback = "yes"
        data = [name,email, text,res, feedback]
        columns=['Name', 'Email', 'Question','res', 'feedback']
        df = pd.DataFrame(dict(zip(columns,data)).values()).T
        df.to_csv('data_feedback.csv', index=False, mode='a',header=False)


    else:
        st.header("Please enter name and email. It will help this bot to improve.")


def production_mode():
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    a[class^="viewerBadge_container*"]  {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    return


def write_header(image):
    # st.image(image, use_column_width=True)
    st.title('Abhishek Nagarjuna')
    st.markdown('''
        Want to know more about me, type your query below:
    ''')

def write_footer():
    st.text('''
       © Copyright 2021, Abhishek nagarjuna.
    ''')

def write_video():
    st.header("Introduction video")
    video_file = open(os.path.join("video/NLP-demo-v5_audio.mp4"), 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)


def search_pipeline(utt):
    document_store = InMemoryDocumentStore()
    text_file = open("AN.txt", "r")
    lines = text_file.readlines()
    dicts = [{"text" :x} for x in lines]
    document_store.write_documents(dicts)
    retriever = TfidfRetriever(document_store=document_store)
    reader = FARMReader("deepset/roberta-base-squad2")
    # Extractive QA
    # pipe_es = Pipeline()
    # pipe_es.add_node(component=retriever, name="es", inputs=["Query"])
    # res = pipe_es.run(query=utt, top_k_retriever=2)
    qa_pipe = ExtractiveQAPipeline(reader=reader, retriever=retriever)
    res = qa_pipe.run(query=utt, top_k_retriever=2, top_k_reader=1)
    # return [x.to_dict()['text'] for x in res["documents"]]
    return res


if __name__ == "__main__":
    image = Image.open("abhi.jpg")
    st.set_page_config(page_title='Abhishek Nagarjuna', page_icon=image)
    # st.title("Course5 AI Labs Text analysis")
    production_mode()
    
    write_header(image)
    # write_video()
    write_ui()
    write_footer()
