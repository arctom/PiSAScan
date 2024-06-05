import streamlit as st # type: ignore
import pandas as pd
import numpy as np
from annotated_text import annotated_text # type: ignore
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px # type: ignore
from collections import Counter
from codigo import process_file

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

st.header("Annotated CV")
with st.container(border=True):
    annotated_text(
    "This ",
    ("is", "verb"),
    " some ",
    ("annotated", "adj"),
    ("text", "noun"),
    " for those of ",
    ("you", "pronoun"),
    " who ",
    ("like", "verb"),
    " this sort of ",
    ("thing", "noun"),
    ".",
    "This ",
    ("is", "verb"),
    " some ",
    ("annotated", "adj"),
    ("text", "noun"),
    " for those of ",
    ("you", "pronoun"),
    " who ",
    ("like", "verb"),
    " this sort of ",
    ("thing", "noun"),
    ".",
    "This ",
    ("is", "verb"),
    " some ",
    ("annotated", "adj"),
    ("text", "noun"),
    " for those of ",
    ("you", "pronoun"),
    " who ",
    ("like", "verb"),
    " this sort of ",
    ("thing", "noun"),
    ".",
    "This ",
    ("is", "verb"),
    " some ",
    ("annotated", "adj"),
    ("text", "noun"),
    " for those of ",
    ("you", "pronoun"),
    " who ",
    ("like", "verb"),
    " this sort of ",
    ("thing", "noun"),
    ".",
    "This ",
    ("is", "verb"),
    " some ",
    ("annotated", "adj"),
    ("text", "noun"),
    " for those of ",
    ("you", "pronoun"),
    " who ",
    ("like", "verb"),
    " this sort of ",
    ("thing", "noun"),
    ".",
    "This ",
    ("is", "verb"),
    " some ",
    ("annotated", "adj"),
    ("text", "noun"),
    " for those of ",
    ("you", "pronoun"),
    " who ",
    ("like", "verb"),
    " this sort of ",
    ("thing", "noun"),
    ".",
    "This ",
    ("is", "verb"),
    " some ",
    ("annotated", "adj"),
    ("text", "noun"),
    " for those of ",
    ("you", "pronoun"),
    " who ",
    ("like", "verb"),
    " this sort of ",
    ("thing", "noun"),
    ".",
    "This ",
    ("is", "verb"),
    " some ",
    ("annotated", "adj"),
    ("text", "noun"),
    " for those of ",
    ("you", "pronoun"),
    " who ",
    ("like", "verb"),
    " this sort of ",
    ("thing", "noun"),
    ".",
    "This ",
    ("is", "verb"),
    " some ",
    ("annotated", "adj"),
    ("text", "noun"),
    " for those of ",
    ("you", "pronoun"),
    " who ",
    ("like", "verb"),
    " this sort of ",
    ("thing", "noun"),
    ".",
    "This ",
    ("is", "verb"),
    " some ",
    ("annotated", "adj"),
    ("text", "noun"),
    " for those of ",
    ("you", "pronoun"),
    " who ",
    ("like", "verb"),
    " this sort of ",
    ("thing", "noun"),
    ".",
    "This ",
    ("is", "verb"),
    " some ",
    ("annotated", "adj"),
    ("text", "noun"),
    " for those of ",
    ("you", "pronoun"),
    " who ",
    ("like", "verb"),
    " this sort of ",
    ("thing", "noun"),
    ".",
    "This ",
    ("is", "verb"),
    " some ",
    ("annotated", "adj"),
    ("text", "noun"),
    " for those of ",
    ("you", "pronoun"),
    " who ",
    ("like", "verb"),
    " this sort of ",
    ("thing", "noun"),
    ".",
    "This ",
    ("is", "verb"),
    " some ",
    ("annotated", "adj"),
    ("text", "noun"),
    " for those of ",
    ("you", "pronoun"),
    " who ",
    ("like", "verb"),
    " this sort of ",
    ("thing", "noun"),
    ".",
    "This ",
    ("is", "verb"),
    " some ",
    ("annotated", "adj"),
    ("text", "noun"),
    " for those of ",
    ("you", "pronoun"),
    " who ",
    ("like", "verb"),
    " this sort of ",
    ("thing", "noun"),
    ".",
    )

PLOT_HEIGHT = 400
PLOT_WIDTH = 400

col1, col2 = st.columns(2)

with col1:
    st.header("Classification")
    chart_data = pd.DataFrame(np.random.rand(3, 3), columns=["Director", "Gerente", "Especialista"])
    random_numbers = np.random.rand(3)
    normalized_random_numbers = random_numbers / np.sum(random_numbers)
    chart_data['Normalized'] = normalized_random_numbers
    fig = px.pie(values=normalized_random_numbers, names=["Director", "Gerente", "Especialista"], height=PLOT_HEIGHT, width=PLOT_WIDTH)
    st.plotly_chart(fig)

with col2:
    st.header("Top Semantic Meaning")
    chart_data = pd.DataFrame(np.random.rand(7, 3), columns=["Director", "Gerente", "Especialista"])
    st.bar_chart(chart_data)

st.header("Semantic Comparison")

col3, col4 = st.columns(2)

with col3:
    st.header("Cloud of words")
    text = 'Fun, fun, awesome, awesome, tubular, astounding, superb, great, amazing, amazing, amazing, amazing'
    wordcloud = WordCloud().generate(text)
    fig, ax = plt.subplots(figsize=(PLOT_WIDTH/100, PLOT_HEIGHT/100))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

with col4:
    st.header("Frequency of words")
    words = np.random.choice(['word1', 'word2', 'word3', 'word4', 'word5'], size=100, replace=True)
    word_counts = Counter(words)
    word_freq_df = pd.DataFrame.from_dict(word_counts, orient='index', columns=['count']).reset_index()
    word_freq_df = word_freq_df.rename(columns={'index': 'word'})
    fig = px.bar(word_freq_df, x='word', y='count', title='Frequency of Words', height=PLOT_HEIGHT, width=PLOT_WIDTH)
    st.plotly_chart(fig)

# st.header("Frequency of words comparisson")

st.markdown(
    "More info at [github.com/arctom/PiSAScan](https://github.com/arctom/PiSAScan)"
)

def configure_sidebar() -> None:
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
            bytes_data = uploaded_file.read()
            st.write("filename:", uploaded_file.name)
            st.write(bytes_data)
            process_file(bytes_data)

        st.divider()

def main():
    """
    Main function to run the Streamlit application
    This function initializes the sidebar and the main page layout.
    """
    configure_sidebar()

if __name__ == "__main__":
    main()