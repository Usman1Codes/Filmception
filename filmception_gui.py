import streamlit as st
import json
import os
from pathlib import Path
import base64
from googletrans import Translator
from gtts import gTTS
import time
import tempfile

# Import for BERT model
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertModel
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda
from tensorflow.keras.models import Model
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

# Optional speech-to-text backends
try:
    import whisper
    WHISPER_AVAILABLE = True
except Exception:
    WHISPER_AVAILABLE = False

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except Exception:
    SR_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="Filmception",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-bottom: 1rem;
    }
    .language-selector {
        padding: 0.5rem 0;
        margin-bottom:  transform: translateY(-1px);
    }
    .movie-container {
        padding: 1rem 0;
        margin-bottom: 1.5rem;
    }
    .movie-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #42a5f5;
    }
    .section-title {
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        color: #90caf9;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #555;
        font-size: 0.8rem;
    }
    .stAudio {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Paths (configurable via env vars for containerized deployments)
TRANSLATION_DIR = Path(os.getenv("FILMCEPTION_TRANSLATION_DIR", str(Path(__file__).parent / "translation")))
AUDIO_DIR = Path(os.getenv("FILMCEPTION_AUDIO_DIR", str(Path(__file__).parent / "audio")))
ALL_TRANSLATIONS_FILE = TRANSLATION_DIR / "all_translations.json"
MODEL_DIR = Path(os.getenv("FILMCEPTION_MODEL_DIR", str(Path(__file__).parent / "model")))

# Language code mapping for translation and gTTS
language_codes = {
    'English': 'en',
    'Arabic': 'ar',
    'Urdu': 'ur',
    'Korean': 'ko',
    'Spanish': 'es',
    'French': 'fr'
}

# Speech recognition language codes (for SpeechRecognition Google backend)
speech_recognition_lang_codes = {
    'English': 'en-US',
    'Arabic': 'ar-SA',
    'Urdu': 'ur-PK',
    'Korean': 'ko-KR',
    'Spanish': 'es-ES',
    'French': 'fr-FR'
}

# Initialize translator
translator = Translator()

# Function to load translations
@st.cache_data
def load_translations():
    try:
        if not ALL_TRANSLATIONS_FILE.exists():
            return {}
        with open(ALL_TRANSLATIONS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load translations: {e}")
        return {}

# Function to get audio file path for existing translations
def get_audio_file_path(movie_id, language):
    return AUDIO_DIR / language / f"{movie_id}.mp3"

# Function to create an auto-play audio component
def get_audio_html(file_path):
    audio_file = open(file_path, 'rb')
    audio_bytes = audio_file.read()
    audio_file.close()
    audio_base64 = base64.b64encode(audio_bytes).decode()
    return f'''
    <audio controls>
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        Your browser does not support the audio element.
    </audio>
    '''

# Function to translate text with rate limiting
def translate_text(text, target_language):
    try:
        if len(text) > 1000:
            text = text[:1000]
        time.sleep(1)
        return translator.translate(text, dest=target_language).text
    except Exception as e:
        st.error(f"Translation error: {e}")
        return None

# Function to convert text to speech
def text_to_speech(text, language_code):
    try:
        tts = gTTS(text=text, lang=language_code, slow=False)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        return temp_file.name
    except Exception as e:
        st.error(f"Error generating audio: {e}")
        return None

# Helper to persist uploaded files temporarily
def save_uploaded_file_to_temp(uploaded_file, suffix=None):
    try:
        suffix = suffix or os.path.splitext(uploaded_file.name)[1]
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(uploaded_file.read())
        temp_file.flush()
        temp_file.close()
        return temp_file.name
    except Exception as e:
        st.error(f"Failed to persist uploaded file: {e}")
        return None

# Transcribe audio using Whisper if available, otherwise SpeechRecognition (WAV/AIFF/FLAC only)
def transcribe_audio_file(file_path, source_language=None):
    try:
        if WHISPER_AVAILABLE:
            # Prefer compact model for latency
            model = whisper.load_model("base")
            whisper_lang = language_codes.get(source_language) if source_language else None
            result = model.transcribe(file_path, language=whisper_lang)
            return result.get('text', '').strip()
        elif SR_AVAILABLE:
            if not file_path.lower().endswith((".wav", ".aiff", ".aif", ".flac")):
                raise RuntimeError("SpeechRecognition requires WAV/AIFF/FLAC. Use one of these formats or install 'openai-whisper'.")
            recognizer = sr.Recognizer()
            with sr.AudioFile(file_path) as source:
                audio = recognizer.record(source)
            sr_lang = speech_recognition_lang_codes.get(source_language, 'en-US')
            return recognizer.recognize_google(audio, language=sr_lang)
        else:
            raise RuntimeError("No speech-to-text backend available. Install 'openai-whisper' or 'SpeechRecognition'.")
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return None

# Custom F1 Score metric class for model reconstruction
class MultiLabelF1Score(tf.keras.metrics.Metric):
    def __init__(self, num_classes, average='micro', threshold=0.5, name='f1_score', **kwargs):
        super(MultiLabelF1Score, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.average = average
        self.threshold = threshold
        self.true_positives = self.add_weight(name='tp', shape=(num_classes,), initializer='zeros')
        self.false_positives = self.add_weight(name='fp', shape=(num_classes,), initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', shape=(num_classes,), initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert probabilities to binary predictions using threshold
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        
        # Calculate TP, FP, FN per class
        tp = tf.reduce_sum(y_true * y_pred, axis=0)
        fp = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
        fn = tf.reduce_sum(y_true * (1 - y_pred), axis=0)
        
        # Update state variables
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)
    
    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
        f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
        
        if self.average == 'micro':
            return tf.reduce_sum(f1) / tf.cast(tf.reduce_sum(tf.cast(tf.math.is_finite(f1), tf.float32)), tf.float32)
        elif self.average == 'macro':
            return tf.reduce_mean(tf.boolean_mask(f1, tf.math.is_finite(f1)))
        elif self.average == 'weighted':
            # Weighted by support (true positives + false negatives)
            weights = (self.true_positives + self.false_negatives) / tf.reduce_sum(self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
            return tf.reduce_sum(f1 * weights)
        else:
            return f1
    
    def reset_state(self):
        self.true_positives.assign(tf.zeros_like(self.true_positives))
        self.false_positives.assign(tf.zeros_like(self.false_positives))
        self.false_negatives.assign(tf.zeros_like(self.false_negatives))

# Load the BERT model and tokenizer for genre prediction
@st.cache_resource
def load_genre_prediction_model():
    try:
        # Initialize the genre names exactly as used in training to ensure correct mapping
        # This list matches the training data mapping for accurate prediction interpretation
        genre_names = [
            'Action', 'ActionAdventure', 'Adventure', 'Animation', 'Blackandwhite',
            'Bollywood', 'Comedy', 'Comedy-drama', 'Comedyfilm', 'CrimeFiction',
            'CrimeThriller', 'Documentary', 'Drama', 'FamilyFilm', 'Fantasy',
            'Filmadaptation', 'Horror', 'Indie', 'JapaneseMovies', 'Musical', 'Mystery',
            'Periodpiece', 'Psychologicalthriller', 'RomanceFilm', 'Romanticcomedy',
            'Romanticdrama', 'ScienceFiction', 'ShortFilm', 'Silentfilm', 'Thriller',
            'Warfilm', 'Western', 'Worldcinema'
        ]
        num_genres = len(genre_names)  # This should be 33 to match the saved weights
        
        # Load the tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Constants for model creation
        MAX_LENGTH = 256
        
        # Recreate the model architecture
        # 1. Load pre-trained DistilBERT model
        base_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # 2. Define input layers
        input_ids_layer = Input(shape=(MAX_LENGTH,), dtype=tf.int32, name='input_ids')
        attention_mask_layer = Input(shape=(MAX_LENGTH,), dtype=tf.int32, name='attention_mask')
        
        # 3. Define the Lambda layer function
        def distilbert_call(inputs):
            input_ids, attention_mask = inputs
            return base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        
        # 4. Create the Lambda layer
        cls_token_output = Lambda(distilbert_call, output_shape=(base_model.config.hidden_size,))([input_ids_layer, attention_mask_layer])
        
        # 5. Add the classification head
        x = Dropout(0.2)(cls_token_output)
        output_layer = Dense(num_genres, activation='sigmoid', name='output_genres')(x)
        
        # 6. Construct the model
        model = Model(inputs=[input_ids_layer, attention_mask_layer], outputs=output_layer)
        
        # 7. Load weights from the saved model
        # Note: This assumes the weights file exists in the MODEL_DIR
        weights_path = MODEL_DIR / "model.weights.h5"
        if weights_path.exists():
            model.load_weights(str(weights_path))
        else:
            raise FileNotFoundError(f"Model weights file not found at {weights_path}")
        
        # Initialize the MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        mlb.fit([genre_names])
        
        return tokenizer, model, mlb
    except Exception as e:
        st.error(f"Error loading genre prediction model: {e}")
        return None, None, None

# Function to predict genres from a movie summary
def predict_genres_for_summary(summary_text, tokenizer, model, mlb_instance, max_length=256, threshold=0.5):
    try:
        # 1. Preprocess (Tokenize)
        tokenized_input = tokenizer(
            [summary_text],  # Needs to be a list
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='tf'
        )
        predict_payload = {
            'input_ids': tokenized_input['input_ids'],
            'attention_mask': tokenized_input['attention_mask']
        }
        
        # 2. Predict probabilities
        probabilities = model.predict(predict_payload)
        
        # 3. Apply threshold
        binary_predictions = (probabilities > threshold).astype(int)
        
        # 4. Convert back to genre names
        predicted_genre_list = mlb_instance.inverse_transform(binary_predictions)
        
        # 5. Return genres and probabilities for debugging
        predicted_genres = predicted_genre_list[0]  # Return the list of genres for the single summary
        genre_probs = dict(zip(mlb_instance.classes_, probabilities[0]))
        return predicted_genres, genre_probs
    except Exception as e:
        st.error(f"Error predicting genres: {e}")
        return [], {}

# Main application
def main():
    # Header
    st.markdown('<div class="main-header">🎬 Filmception</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multilingual Movie Summary Translator</div>', unsafe_allow_html=True)

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Existing Summaries", "Live Summary Conversion", "Genre Prediction", "Upload, Translate & TTS"])

    # Tab 1: Existing Summaries (original functionality)
    with tab1:
        # Load translations
        translations = load_translations()
        if not translations:
            st.info("No preloaded translations found. Use the Upload tab or Live Conversion to generate content.")
        else:
            movie_ids = list(translations.keys())
            
            # Sidebar
            with st.sidebar:
                st.markdown("## Movie Selection")
                st.markdown("Select a movie to view its translated summaries and listen to audio.")
                
                selected_movie_index = st.selectbox(
                    "Choose a movie:",
                    options=range(len(movie_ids)),
                    format_func=lambda x: f"Movie {x+1}: {translations[movie_ids[x]]['original'][:50]}...",
                )
                
                selected_movie_id = movie_ids[selected_movie_index]
                
                st.markdown("---")
                
                st.markdown("## Language Settings")
                st.markdown("Select languages for translation and audio playback.")
                
                translation_language = st.radio(
                    "View translation in:",
                    options=["Original", "Arabic", "Urdu", "Korean"],
                    index=0,
                )
                
                audio_language = st.radio(
                    "Play audio in:",
                    options=["Arabic", "Urdu", "Korean"],
                    index=0,
                )
                
                st.markdown("---")
                st.markdown("## About Filmception")
                st.markdown("""
                Filmception is an AI-powered multilingual movie summary 
                translator and genre classifier built for the Artificial 
                Intelligence Spring 2025 Semester Project.
                """)
            
            # Main content
            selected_movie = translations[selected_movie_id]
            
            st.markdown('<div class="movie-container">', unsafe_allow_html=True)
            
            st.markdown(f'<div class="movie-title">Movie ID: {selected_movie_id}</div>', unsafe_allow_html=True)
            
            if translation_language == "Original":
                st.markdown('<div class="section-title">Original Summary:</div>', unsafe_allow_html=True)
                st.markdown(f"<p>{selected_movie['original']}</p>", unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="section-title">{translation_language} Translation:</div>', unsafe_allow_html=True)
                st.markdown(f"<p>{selected_movie['translations'][translation_language]}</p>", unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="movie-container">', unsafe_allow_html=True)
            st.markdown(f'<div class="section-title">Audio Playback - {audio_language}</div>', unsafe_allow_html=True)
            
            audio_file_path = get_audio_file_path(selected_movie_id, audio_language)
            
            if os.path.exists(audio_file_path):
                st.audio(str(audio_file_path))
            else:
                st.warning(f"Audio file not found for Movie ID {selected_movie_id} in {audio_language}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            with st.expander("Additional Features", expanded=False):
                st.markdown('<div class="section-title">Compare All Translations</div>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("<b>Arabic</b>", unsafe_allow_html=True)
                    st.markdown(f"<p>{selected_movie['translations']['Arabic']}</p>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<b>Urdu</b>", unsafe_allow_html=True)
                    st.markdown(f"<p>{selected_movie['translations']['Urdu']}</p>", unsafe_allow_html=True)
                
                with col3:
                    st.markdown("<b>Korean</b>", unsafe_allow_html=True)
                    st.markdown(f"<p>{selected_movie['translations']['Korean']}</p>", unsafe_allow_html=True)
                
                st.markdown('<div class="section-title">Compare All Audio Files</div>', unsafe_allow_html=True)
                
                audio_col1, audio_col2, audio_col3 = st.columns(3)
                
                with audio_col1:
                    st.markdown("<b>Arabic Audio</b>", unsafe_allow_html=True)
                    arabic_audio_path = get_audio_file_path(selected_movie_id, "Arabic")
                    if os.path.exists(arabic_audio_path):
                        st.audio(str(arabic_audio_path))
                    else:
                        st.warning("Arabic audio not found")
                
                with audio_col2:
                    st.markdown("<b>Urdu Audio</b>", unsafe_allow_html=True)
                    urdu_audio_path = get_audio_file_path(selected_movie_id, "Urdu")
                    if os.path.exists(urdu_audio_path):
                        st.audio(str(urdu_audio_path))
                    else:
                        st.warning("Urdu audio not found")
                
                with audio_col3:
                    st.markdown("<b>Korean Audio</b>", unsafe_allow_html=True)
                    korean_audio_path = get_audio_file_path(selected_movie_id, "Korean")
                    if os.path.exists(korean_audio_path):
                        st.audio(str(korean_audio_path))
                    else:
                        st.warning("Korean audio not found")

    # Tab 2: Live Summary Conversion
    with tab2:
        st.markdown("## Live Summary Conversion")
        st.markdown("Enter a movie summary and convert it to translated text and audio in the selected language.")

        # User input for movie summary
        user_summary = st.text_area("Enter Movie Summary:", height=150)

        # Single language selection for both translation and audio
        conversion_language = st.selectbox(
            "Select Conversion Language:",
            options=["English", "Arabic", "Urdu", "Korean", "Spanish", "French"],
            index=0
        )

        # Button to trigger translation and audio generation
        if st.button("Convert Summary"):
            if user_summary.strip():
                # Translate the summary
                st.markdown('<div class="section-title">Translated Summary</div>', unsafe_allow_html=True)
                translated_text = translate_text(user_summary, language_codes[conversion_language])
                if translated_text:
                    st.markdown(f"<p>{translated_text}</p>", unsafe_allow_html=True)
                else:
                    st.warning("Failed to translate the summary.")

                # Generate and play audio
                st.markdown(f'<div class="section-title">Audio Playback - {conversion_language}</div>', unsafe_allow_html=True)
                audio_file = text_to_speech(translated_text, language_codes[conversion_language])
                if audio_file:
                    st.audio(audio_file)
                    os.unlink(audio_file)  # Clean up temporary file
                else:
                    st.warning("Failed to generate audio.")
            else:
                st.warning("Please enter a movie summary.")

    # Tab 3: Genre Prediction (Implementation)
    with tab3:
        st.markdown("## Genre Prediction")
        st.markdown("Enter a movie summary to predict its genre(s) using our BERT-based model.")
        
        # Load the model and tokenizer
        tokenizer, model, mlb = load_genre_prediction_model()
        
        if tokenizer is None or model is None or mlb is None:
            st.error("Failed to load the genre prediction model. Please check the model files.")
        else:
            # User input for movie summary
            genre_summary = st.text_area("Enter Movie Summary for Genre Prediction:", height=150)
            
            # Configure prediction threshold
            threshold = st.slider(
                "Prediction Confidence Threshold", 
                min_value=0.1, 
                max_value=0.9, 
                value=0.5,  
                step=0.1,
                help="Lower values will predict more genres, higher values will be more selective. Default set to 0.5 as used in model evaluation."
            )
            
            # Button to trigger genre prediction
            if st.button("Predict Genres"):
                if genre_summary.strip():
                    # Show a spinner while predicting
                    with st.spinner("Predicting genres..."): 
                        # Make prediction
                        predicted_genres, genre_probs = predict_genres_for_summary(
                            genre_summary, 
                            tokenizer, 
                            model, 
                            mlb, 
                            threshold=threshold
                        )
                    
                    # Display results
                    st.markdown('<div class="section-title">Predicted Genres</div>', unsafe_allow_html=True)
                    
                    if predicted_genres and len(predicted_genres) > 0:
                        # Create color-coded genre badges
                        genre_html = ""
                        genre_colors = {
                            'Action': '#E73',      # Light Red
                            'ActionAdventure': '#E73',
                            'Adventure': '#81C784',   # Light Green
                            'Animation': '#64B5F6',   # Light Blue
                            'Blackandwhite': '#B0BEC5',
                            'Bollywood': '#FFD54F',
                            'Comedy': '#FFD54F',      # Light Amber
                            'Comedy-drama': '#FFD54F',
                            'Comedyfilm': '#FFD54F',
                            'CrimeFiction': '#4DD0E1',# Light Cyan
                            'CrimeThriller': '#4DD0E1',
                            'Documentary': '#A1887F', # Light Brown
                            'Drama': '#FF8A65',       # Light Deep Orange
                            'FamilyFilm': '#F06292',  # Light Pink
                            'Fantasy': '#7986CB',     # Light Indigo
                            'Filmadaptation': '#9575CD',
                            'Horror': '#5C6BC0',      # Light Indigo
                            'Indie': '#A1887F',
                            'JapaneseMovies': '#FFB74D',
                            'Musical': '#FFF176',     # Light Yellow
                            'Mystery': '#9575CD',     # Light Deep Purple
                            'Periodpiece': '#4FC3F7',
                            'Psychologicalthriller': '#FF8A65',
                            'RomanceFilm': '#F48FB1', # Light Pink
                            'Romanticcomedy': '#F48FB1',
                            'Romanticdrama': '#F48FB1',
                            'ScienceFiction': '#4DD0E1', # Light Cyan
                            'ShortFilm': '#64B5F6',
                            'Silentfilm': '#B0BEC5',
                            'Thriller': '#FF8A65',    # Light Deep Orange
                            'Warfilm': '#A5D6A7',     # Light Green
                            'Western': '#FFB74D',     # Light Orange
                            'Worldcinema': '#7986CB'
                        }
                        
                        for genre in predicted_genres:
                            color = genre_colors.get(genre, '#90A4AE')  # Default color if genre not in dictionary
                            genre_html += f'<span style="display: inline-block; background-color: {color}; color: white; padding: 5px 10px; margin: 5px; border-radius: 15px; font-weight: bold;">{genre}</span>'
                        
                        st.markdown(f"<div style='margin: 20px 0;'>{genre_html}</div>", unsafe_allow_html=True)
                        
                        # Debug information: Show probabilities for predicted genres
                        with st.expander("View Prediction Probabilities (Debug Info)"):
                            st.markdown("**Confidence Scores for Predicted Genres:**")
                            prob_table = "| Genre | Confidence Score |\n|-------|------------------|\n"
                            for genre in predicted_genres:
                                prob = genre_probs.get(genre, 0.0)
                                prob_table += f"| {genre} | {prob:.3f} |\n"
                            st.markdown(prob_table)
                    else:
                        st.warning("No genres predicted. Try lowering the confidence threshold or providing a more detailed summary.")
                        
                        # Debug information: Show top probabilities even if no genres predicted
                        with st.expander("View Top Prediction Probabilities (Debug Info)"):
                            st.markdown("**Top 5 Confidence Scores (Even if Below Threshold):**")
                            if genre_probs:
                                sorted_probs = sorted(genre_probs.items(), key=lambda x: x[1], reverse=True)[:5]
                                prob_table = "| Genre | Confidence Score |\n|-------|------------------|\n"
                                for genre, prob in sorted_probs:
                                    prob_table += f"| {genre} | {prob:.3f} |\n"
                                st.markdown(prob_table)
                            else:
                                st.markdown("No probability data available.")
                else:
                    st.warning("Please enter a movie summary.")
                    
    # Tab 4: Upload, Translate & TTS
    with tab4:
        st.markdown("## Upload, Translate & TTS")
        upload_mode = st.radio("What would you like to upload?", options=["Audio file", "Text summary"], index=0)

        if upload_mode == "Audio file":
            uploaded_audio = st.file_uploader(
                "Upload an audio file (Whisper supports mp3/m4a/mp4/wav/flac; SpeechRecognition supports wav/aiff/flac)",
                type=["mp3", "m4a", "mp4", "wav", "flac", "aiff", "aif", "ogg"]
            )
            source_language = st.selectbox(
                "Source language (optional)",
                options=list(language_codes.keys()),
                index=0,
                help="Used to guide STT. Leave as English or pick the actual language if known."
            )
            target_languages = st.multiselect(
                "Target languages for translation",
                options=list(language_codes.keys()),
                default=["English", "Arabic", "Urdu", "Korean"]
            )
            generate_audio = st.checkbox("Generate audio for translated text", value=True)

            if st.button("Transcribe and Translate"):
                if uploaded_audio is None:
                    st.warning("Please upload an audio file.")
                else:
                    temp_path = save_uploaded_file_to_temp(uploaded_audio)
                    if temp_path:
                        with st.spinner("Transcribing audio..."):
                            transcript = transcribe_audio_file(temp_path, source_language=source_language)
                        os.unlink(temp_path)

                        if transcript and transcript.strip():
                            st.markdown('<div class="section-title">Transcript</div>', unsafe_allow_html=True)
                            st.text_area("Transcribed Text", transcript, height=150)

                            st.markdown('<div class="section-title">Translations</div>', unsafe_allow_html=True)
                            for lang in target_languages:
                                st.markdown(f"**{lang}**")
                                translated_text = translate_text(transcript, language_codes[lang])
                                if translated_text:
                                    st.markdown(f"<p>{translated_text}</p>", unsafe_allow_html=True)
                                    if generate_audio:
                                        audio_file = text_to_speech(translated_text, language_codes[lang])
                                        if audio_file:
                                            st.audio(audio_file)
                                            os.unlink(audio_file)
                                        else:
                                            st.warning(f"Failed to generate audio for {lang}.")
                                else:
                                    st.warning(f"Failed to translate to {lang}.")
                        else:
                            st.warning("No transcript produced. Check the audio quality or backend availability.")

        else:  # Text summary
            uploaded_text = st.file_uploader("Upload a text file (.txt) containing the summary", type=["txt"])
            manual_text = st.text_area("Or paste a summary here:", height=150)
            target_languages = st.multiselect(
                "Target languages for translation",
                options=list(language_codes.keys()),
                default=["English", "Arabic", "Urdu", "Korean"]
            )
            generate_audio = st.checkbox("Generate audio for translated text", value=True)

            if st.button("Translate Text"):
                text_content = None
                if uploaded_text is not None:
                    try:
                        text_content = uploaded_text.read().decode("utf-8").strip()
                    except Exception:
                        st.error("Failed to read uploaded file as UTF-8.")
                        text_content = None
                if not text_content:
                    text_content = manual_text.strip()

                if not text_content:
                    st.warning("Please upload a text file or paste some text.")
                else:
                    st.markdown('<div class="section-title">Original Text</div>', unsafe_allow_html=True)
                    st.text_area("Original", text_content, height=150)

                    st.markdown('<div class="section-title">Translations</div>', unsafe_allow_html=True)
                    for lang in target_languages:
                        st.markdown(f"**{lang}**")
                        translated_text = translate_text(text_content, language_codes[lang])
                        if translated_text:
                            st.markdown(f"<p>{translated_text}</p>", unsafe_allow_html=True)
                            if generate_audio:
                                audio_file = text_to_speech(translated_text, language_codes[lang])
                                if audio_file:
                                    st.audio(audio_file)
                                    os.unlink(audio_file)
                                else:
                                    st.warning(f"Failed to generate audio for {lang}.")
                        else:
                            st.warning(f"Failed to translate to {lang}.")
                    
    # Footer
    st.markdown('<div class="footer">Filmception - AI-powered Multilingual Movie Summary Translator and Genre Classifier</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()