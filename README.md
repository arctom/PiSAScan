# CV Classifier

This is a simple Streamlit application for classifying Curriculum Vitae (CVs) into different categories such as Director, Manager, and Specialist based on their characteristics and content.

## Installation

To install the required dependencies, you can use pip and the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Usage

To run the application, use the following command:
    
```bash
streamlit run main.py
```

This will start the Streamlit server and launch the application in your default web browser.

## Application Features

- Annotated CV: View annotated CVs with highlighted semantic elements.
- Classification: Classify CVs into different categories using pie charts and bar charts.
- Semantic Comparison: Compare word clouds and word frequency charts.
- File Upload: Upload CVs for classification.


## Folder Structure
The project folder structure is as follows:

```bash
.
├── main.py                 # Main Streamlit application file
├── process_file.py         # Python file for processing uploaded files
├── requirements.txt         # List of Python dependencies
└── README.md               # This README file
```

## Contribution
Contributions are welcome! If you have suggestions, bug reports, or feature requests, please open an issue or create a pull request.
