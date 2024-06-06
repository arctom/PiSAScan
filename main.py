import streamlit as st
import pandas as pd
import numpy as np
from annotated_text import annotated_text
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
from collections import Counter
from codigo import main_code
# import pyPDF2

BACKGROUND_COLOR = 'white'
COLOR = 'black'

st.set_page_config(
    page_title='CV Classifier',
    layout='wide',
    page_icon='./images/favicon.ico'
    )

def set_page_container_style(
        max_width: int = 1100, max_width_100_percent: bool = False,
        padding_top: int = 1, padding_right: int = 10, padding_left: int = 1, padding_bottom: int = 10,
        color: str = COLOR, background_color: str = BACKGROUND_COLOR,
    ):
        if max_width_100_percent:
            max_width_str = f'max-width: 100%;'
        else:
            max_width_str = f'max-width: {max_width}px;'
        st.markdown(
            f'''
            <style>
                .reportview-container .css-1lcbmhc .css-1outpf7 {{
                    padding-top: 35px;
                }}
                .reportview-container .main .block-container {{
                    {max_width_str}
                    padding-top: {padding_top}rem;
                    padding-right: {padding_right}rem;
                    padding-left: {padding_left}rem;
                    padding-bottom: {padding_bottom}rem;
                }}
                .reportview-container .main {{
                    color: {color};
                    background-color: {background_color};
                }}
            </style>
            ''',
            unsafe_allow_html=True,
        )

# Main Page
st.markdown("""
# CV Classifier
""")

st.markdown(
            f'''
            <style>
                .reportview-container .sidebar-content {{
                    padding-top: {1}rem;
                }}
                .reportview-container .main .block-container {{
                    padding-top: {1}rem;
                }}
            </style>
            ''',unsafe_allow_html=True)

PLOT_HEIGHT = 400
PLOT_WIDTH = 400

# st.header("Frequency of words comparisson")

def configure_sidebar():
    """
    Setup and display the sidebar
    """
    with st.sidebar:

        st.title("CV Classifier")
        st.subheader("A project for Curriculum Vitae classification.")
        st.markdown("""
        This project aims to develop an automated system that classifies resumes (CVs) into different categories such as Director, Manager, and Specialist based on their characteristics and content. This system has been specifically designed to meet the needs of Pisa, helping to optimize the personnel selection process and ensuring that candidates are evaluated efficiently and accurately.
        """)

        st.markdown("""
            <style>
                .custom-file-uploader {
                    border: 2px dashed #aaa;
                    padding: 10px;
                    border-radius: 5px;
                    text-align: center;
                }
                .custom-file-uploader:hover {
                    background-color: #f0f0f0;
                }
            </style>
            <div class="custom-file-uploader">
                <h4>Upload your CV</h4>
                <p>↓ Select a file to upload ↓</p>
            </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("", key="fileUploader")

        st.markdown("""
            <div class="custom-file-uploader">
                <p>Accepted formats: PDF, DOCX, DOC</p>
            </div>
        """, unsafe_allow_html=True)

        if uploaded_file is not None:
            # bytes_data = uploaded_file.read()
            # st.write("filename:", uploaded_file.name)
            # st.write(bytes_data)
            # main_code(bytes_data)
            with open('test.pdf', 'wb') as f:
                f.write(uploaded_file.read())
            return main_code('D:/Github/PiSAScan/test.pdf') 

        st.divider()

def colored_text(X):
    st.header("Annotated CV")
    text = []
    print(X.columns)
    for index, row in X.iterrows():
        mapping = (row['sentence'], str(row['cluster']))
        text.append(mapping)
    with st.container(border=True):
        annotated_text(text)

def predict_proba(X, results):
    resultados = results[0]

    col1, col2 = st.columns(2)

    with col1:
        st.header("Classification")
        chart_data = pd.DataFrame({"Role": ["Gerente", "Especialista", "Director"],"Probability": resultados})
        fig = px.pie(chart_data, values='Probability', names='Role', height=PLOT_HEIGHT, width=PLOT_WIDTH)
        st.plotly_chart(fig)

    with col2:
        st.header("Top Semantic Meaning")
        cluster_counts = X['cluster'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster', 'Count']
        cluster_counts = cluster_counts.sort_values(by='Count', ascending=False).iloc[0:5, :]
        bar_chart = px.bar(cluster_counts, x='Cluster', y='Count', height=PLOT_HEIGHT, width=PLOT_WIDTH)
        st.plotly_chart(bar_chart)

    all_words = " ".join(X['sentence'].to_list())
    col3, col4 = st.columns(2)
    print(all_words)
    with col3:
        st.header("Cloud of words")
        text = all_words
        wordcloud = WordCloud().generate(text)
        fig, ax = plt.subplots(figsize=(PLOT_WIDTH/100, PLOT_HEIGHT/100))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

    with col4:
        st.header("Frequency of words")
        words = all_words.split(" ")
        word_counts = Counter(words)
        word_freq_df = pd.DataFrame.from_dict(word_counts, orient='index', columns=['count']).reset_index()
        print(word_freq_df)
        word_freq_df = word_freq_df.sort_values(by='count', ascending=False).iloc[0:5, :]
        word_freq_df = word_freq_df.rename(columns={'index': 'word'})
        fig = px.bar(word_freq_df, x='word', y='count', height=PLOT_HEIGHT, width=PLOT_WIDTH)
        st.plotly_chart(fig)


def main():
    """
    Main function to run the Streamlit application
    This function initializes the sidebar and the main page layout.
    """
    results, X = configure_sidebar()
    colored_text(X)
    predict_proba(X, results)
    # cloud_of_words(X)
    st.markdown(
    "More info at [github.com/arctom/PiSAScan](https://github.com/arctom/PiSAScan)"
    )

if __name__ == "__main__":
    main()