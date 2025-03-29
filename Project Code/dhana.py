import sys
sys.path.append('/vercel/path0/Project Code')

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, session
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import regex as re
from collections import Counter
import emoji
import pandas as pd
import numpy as np
import neattext.functions as nfx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import contractions
from transformers import pipeline
from googletrans import Translator
import string
from nltk.corpus import stopwords
from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import logging
logging.getLogger('transformers').setLevel(logging.ERROR)

app = Flask(__name__)
app.secret_key = "123"

# In-memory user storage (replace with database in production)
USERS = {
    "admin": {"password": "admin123", "email": "admin@example.com", "address": "123 Main St"},
    "user1": {"password": "password1", "email": "user1@example.com", "address": "456 Oak Ave"}
}

# Initialize analyzers
sid = SentimentIntensityAnalyzer()
model_sa = pipeline("sentiment-analysis")

# Chat analysis functions
def date_time(s):
    pattern = r'^([0-9]+)(\/)([0-9]+)(\/)([0-9]+), ([0-9]+):([0-9]+)[ ]?(AM|PM|am|pm)? -'
    return bool(re.match(pattern, s))

def find_contact(s):
    return len(s.split(":")) == 2

def get_message(line):
    split_line = line.split(' - ')
    datetime = split_line[0]
    date, time = datetime.split(', ')
    message = " ".join(split_line[1:])
    if find_contact(message):
        split_message = message.split(": ")
        author = split_message[0]
        message = split_message[1]
    else:
        author = None
    return date, time, author, message

def count_emojis(text):
    return len([char for char in text if char in emoji.UNICODE_EMOJI['en']])

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == 'POST':
        username = request.form['name']
        password = request.form['password']
        if username in USERS and USERS[username]['password'] == password:
            session["username"] = username
            session["email"] = USERS[username]['email']
            return redirect(url_for('home'))
        flash("Username and Password Mismatch", "danger")
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['name']
        if username in USERS:
            flash("Username already exists", "danger")
            return redirect(url_for('register'))
        USERS[username] = {
            "password": request.form['Password'],
            "email": request.form['Email'],
            "address": request.form['address']
        }
        flash("Registration Successful", "success")
        return redirect(url_for('index'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/Frontpage')
def home():
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('Frontpage.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'username' not in session:
        return redirect(url_for('index'))
        
    file = request.files['file']
    text = file.read().decode('utf-8')
    
    # Process chat
    data = []
    buffer = []
    date = time = author = None
    
    for line in text.split('\n'):
        if date_time(line):
            if buffer:
                data.append([date, time, author, ''.join(buffer)])
                buffer.clear()
            date, time, author, message = get_message(line)
            buffer.append(message)
        else:
            buffer.append(line)
            
    if buffer:
        data.append([date, time, author, ''.join(buffer)])
    
    df = pd.DataFrame(data, columns=["Date", "Time", "Contact", "Message"])
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
   
    # Perform analysis
    scores = sid.polarity_scores(text)
    sentiment = scores['compound']
    sentiment_class = 'positive' if sentiment > 0.05 else 'negative' if sentiment < -0.05 else 'neutral'
    
    # Calculate statistics
    emoji_counter = Counter(c for msg in df['Message'] for c in msg if c in emoji.UNICODE_EMOJI['en'])
    top_emojis = emoji_counter.most_common(5)
    
    stats = {
        'sentiment': sentiment,
        'sentiment_class': sentiment_class,
        'num_messages': len(data),
        'media_messages': df[df["Message"]=='<Media omitted>'].shape[0],
        'num_links': len(re.findall(r'(http[s]?://\S+)', text)),
        'total_emojis': sum(count_emojis(msg) for msg in df['Message']),
        'start_date': df['Date'].min().date(),
        'end_date': df['Date'].max().date(),
        'contacts': ', '.join(filter(None, df['Contact'].unique())),
        'num_users': df['Contact'].nunique(),
        'num_messages_per_person': ', '.join(f"{k}: {v}" for k,v in df['Contact'].value_counts().items()),
        'most_active_person': df['Contact'].value_counts().idxmax(),
        'most_common_emoji': top_emojis[0][0] if top_emojis else None,
        'emoji_count': top_emojis[0][1] if top_emojis else 0,
        'num_emojis': len(emoji_counter),
        'top_emojis': top_emojis,
        'busiest_hour': df['Time'].str.extract(r'(\d+):')[0].value_counts().idxmax(),
        'busiest_day': df['Date'].dt.day_name().value_counts().idxmax()
    }
    
    return render_template('result.html', **stats)

# Text analysis routes
@app.route('/sen')
def sentence():
    return render_template('sentence.html')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    tokens = [word for word in nltk.word_tokenize(text) if word not in stopwords.words('english')]
    return " ".join(tokens)

Behaviour_labels = {
    "Crime": ["i will kill you","kill","killed","kill me","killing","i want to kill you"],
    "Short-temper": ["annoyance"],
    "Appreciation": ["gratitude"],
    "Enthusiasm": ["joy", "excitement"],
    "Optimism": ["optimistic"],
    "Caring": ["caring"],
    "Relief": ["relaxation"],
    "Depressed": ["sadness", "grief"],
    "Disgust": ["disgust", "embarrassment"],
    "Introvert": ["nervousness"],
}

def map_Behaviour_label(label):
    for behaviour, keywords in Behaviour_labels.items():
        if any(keyword in label.lower() for keyword in keywords):
            return behaviour
    return None

# Initialize behavior analysis models
try:
    behavior_tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
    behavior_model = TFRobertaForSequenceClassification.from_pretrained("arpanghoshal/EmoRoBERTa")
    behavior_pipeline = pipeline('sentiment-analysis', model=behavior_model, tokenizer=behavior_tokenizer)
    
    behavior_df = pd.read_csv("behaviour.csv")
    behavior_df["Message"] = behavior_df["Message"].apply(preprocess_text)
    
    behavior_vectorizer = TfidfVectorizer()
    X_train = behavior_vectorizer.fit_transform(behavior_df['Message'])
    behavior_svm = LinearSVC().fit(X_train, behavior_df['Behaviour'])
    
except Exception as e:
    print(f"Error loading behavior models: {e}")

def predict_Behaviour_label(text):
    result = behavior_pipeline(text)[0]
    behaviour = map_Behaviour_label(result["label"])
    if behaviour is None:
        text_vectorized = behavior_vectorizer.transform([text])
        behaviour = behavior_svm.predict(text_vectorized)[0]
    return behaviour

@app.route("/pre", methods=["POST","GET"])
def senpre():
    text = request.form['text']
    behaviour = predict_Behaviour_label(text)
    sentiment = "positive" if sid.polarity_scores(text)['compound'] >= 0 else "negative"
    
    # Your behavior-sentiment mapping logic here
    output = f"The sentiment is {sentiment} and behavior is {behaviour}"
    
    return render_template("pre.html", output=output)

@app.route('/voice')
def voice():
    return render_template('voice_input.html')

@app.route("/voicepre", methods=["POST","GET"])
def voicepre():
    text = request.form.get('text', '')
    if not text:
        return render_template('voice_input.html', 
                           prediction='Please type your message as voice input is not supported')
    
    behaviour = predict_Behaviour_label(text)
    sentiment = "positive" if sid.polarity_scores(text)['compound'] >= 0 else "negative"
    
    # Your behavior-sentiment mapping logic here
    output = f"The sentiment is {sentiment} and behavior is {behaviour}"
    
    return render_template("voicepre.html", output=output)

@app.route('/para')
def para():
    return render_template('para.html')

@app.route('/paragraph_sentiment', methods=['POST'])
def paragraph_sentiment():
    paragraph = request.form['paragraph']
    label, translated = get_paragraph_sentiment(paragraph)
    return render_template('paragraph_sentiment.html', 
                         paragraph=paragraph, 
                         label=label, 
                         translated_paragraph=translated)

if __name__ == '__main__':
    app.run(debug=True)
