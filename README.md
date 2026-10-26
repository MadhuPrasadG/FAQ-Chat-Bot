University FAQ Chatbot 🎓🤖

# Project Task:
 Develop a simple chatbot that answers frequently asked questions. (SRN: R25DG012)
Author: MADHU PRASAD G

# Overview
This is a Machine Learning-based mini project that acts as an intelligent FAQ chatbot for university students. It accepts user questions, processes the text, and finds the most relevant pre-stored answer using natural language processing techniques. 

Instead of a basic terminal prompt, this project features a modern, user-friendly web interface powered by Gradio.

## Features
* Text Preprocessing: Cleans user input by standardizing text and removing punctuation using Regular Expressions (Regex).
* TF-IDF Vectorization: Converts text data into numerical vectors using `scikit-learn` so the computer can process the language mathematically.
* Cosine Similarity: Calculates the mathematical distance between the user's query and the stored FAQ database to return the most accurate match.
* Web GUI: A fully interactive chat interface built with `gradio`.

How to Run the Project
1. Clone this repository to your local machine.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt