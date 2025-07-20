# FilmCeption: Movie Summary Translator & Genre Predictor

## 🎬 What is FilmCeption?
FilmCeption is an AI-powered web application that analyzes movie summaries. It can:
1.  **Translate** summaries into multiple languages (Arabic, Urdu, Korean, Spanish, French).
2.  **Generate Text-to-Speech** audio for these translations.
3.  **Predict** the genres of a movie based on its summary using a Deep Learning model (DistilBERT).

It's designed to showcase NLP, translation, TTS, and classification techniques applied to film data.

## ✨ Features
*   Read and listen to summaries in multiple languages.
*   Translate and generate audio for your own summaries in real-time.
*   Get AI-driven genre predictions based on plot descriptions.
*   Interactive interface built with Streamlit.

## ⚙️ Requirements & Setup

### Prerequisites
*   Python 3.8+
*   Git
*   pip (Python package installer)

### Essential Files & Folders
To run the application, you primarily need:
1.  **The Application Script:** `filmception_gui.py`
2.  **The Trained Model:** A `final_model/` directory containing the saved genre prediction model (e.g., `final_model.keras`).
3.  **The Dataset:** A `MovieSummaries/` directory containing the CMU Movie Summary Corpus.

### Setup Steps
1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/FilmCeption.git # Replace with your repo URL
    cd FilmCeption
    ```
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    # Download necessary NLTK data packages
    python -m nltk.downloader punkt stopwords wordnet
    ```
    *(Key dependencies include: `streamlit`, `pandas`, `googletrans==4.0.0-rc1`, `gTTS`, `tensorflow`, `transformers`, `scikit-learn`)*
3.  **Download the Dataset:**
    *   Download the CMU Movie Summary Corpus from Kaggle: [https://www.kaggle.com/datasets/msafi04/movies-genre-dataset-cmu-movie-summary](https://www.kaggle.com/datasets/msafi04/movies-genre-dataset-cmu-movie-summary)
    *   Extract the files (`movie.metadata.tsv`, `plot_summaries.txt`).
    *   Create a directory named `MovieSummaries` in the project root and place these two files inside it.
4.  **Add the Model:**
    *   Ensure you have the trained genre model files.
    *   Create a directory named `final_model` in the project root and place the model files (e.g., `final_model.keras`) inside it.
5.  **(Optional) Add Pre-processed Data:**
    *   If you have pre-generated translations (`all_translations.json` etc.), place them in a `translation/` directory.
    *   If you have pre-generated audio files (`.mp3`), place them in language subfolders within an `Audiofiles/` directory. *The application can still perform live translation/TTS without these.*

## ▶️ How to Run
1.  Make sure you are in the project's root directory (`FilmCeption/`) in your terminal.
2.  Execute the Streamlit command:
    ```bash
    streamlit run filmception_gui.py
    ```
3.  Your web browser should automatically open to the application's interface (typically `http://localhost:8501`).
4.  Use the tabs ("Existing Summaries", "Live Summary Conversion", "Genre Prediction") to interact with the different features.


![alt text](<Screenshot from 2025-07-20 14-05-57.png>) ![alt text](<Screenshot from 2025-07-20 14-01-29.png>) ![alt text](<Screenshot from 2025-07-20 14-00-36.png>)