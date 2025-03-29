# In your app.py, add this at the top:
import sys
sys.path.append('/vercel/path0/Project Code')

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, session
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import regex as re
from collections import Counter
import emoji
import matplotlib.pyplot as plt
import seaborn as sns
import io
from io import BytesIO
import tensorflow as tf
import torch
import base64
import time
import pandas as pd
import numpy as np
import os
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
app.secret_key="123"


# Dummy user data - replace with your actual user management system
USERS = {
    "admin": {"password": "admin123", "email": "admin@example.com", "address": "123 Main St"},
    "user1": {"password": "password1", "email": "user1@example.com", "address": "456 Oak Ave"}
}


# Extract the Date time
def date_time(s):
    pattern = r'^([0-9]+)(\/)([0-9]+)(\/)([0-9]+), ([0-9]+):([0-9]+)[ ]?(AM|PM|am|pm)? -'
    result = re.match(pattern, s)
    if result:
        return True
    return False

# Extract contacts
def find_contact(s):
    s = s.split(":")
    if len(s) == 2:
        return True
    else:
        return False

# Extract Message
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
#emoji
def count_emojis(text):
    emoji_list = [char for char in text if char in emoji.UNICODE_EMOJI]
    return len(emoji_list)


#speech to text
def convert_speech_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak anything...")
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        print("You said: {}".format(text))
        return text
    except:
        print("Sorry, could not recognize your voice")
        return None

    

# Load the sentiment analysis model
model_sa = pipeline("sentiment-analysis")

# Define a function to get the sentiment of a paragraph
def get_paragraph_sentiment(paragraph):
    # Translate the paragraph to English
    translator = Translator()
    translated_paragraph = translator.translate(paragraph, dest='en').text

    # Use the model to predict the sentiment label and score for the paragraph
    result = model_sa(translated_paragraph)[0]
    label = result['label']
    score = result['score']

    # Return the sentiment label and the translated paragraph
    return label, translated_paragraph

# Load and preprocess the data
df = pd.read_csv("Tweets.csv")
review_df = df[['text', 'airline_sentiment']]
review_df = review_df[review_df['airline_sentiment'] != 'neutral']

# Load the pre-trained model
model = load_model('sentiment_model.h5')

# Tokenize the text
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(review_df['text'])


#audio


# initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# a function that splits the audio file into chunks
# and applies speech recognition
def get_large_audio_transcription(path):
    """
    Splitting the large audio file into chunks
    and apply speech recognition on each of these chunks
    """
    # create a speech recognition object
    r = sr.Recognizer()
    # open the audio file using pydub
    sound = AudioSegment.from_wav(path)  
    # split audio sound where silence is 700 miliseconds or more and get chunks
    chunks = split_on_silence(sound,
        # experiment with this value for your target audio file
        min_silence_len = 500,
        # adjust this per requirement
        silence_thresh = sound.dBFS-14,
        # keep the silence for 1 second, adjustable as well
        keep_silence=500,
    )
    folder_name = "audio-chunks"
    # create a directory to store the audio chunks
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    whole_text = ""
    # process each chunk 
    for i, audio_chunk in enumerate(chunks, start=1):
        # export audio chunk and save it in
        # the `folder_name` directory.
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")
        # recognize the chunk
        with sr.AudioFile(chunk_filename) as source:
            audio_listened = r.record(source)
            # try converting it to text
            try:
                text = r.recognize_google(audio_listened)
            except sr.UnknownValueError as e:
                print("Error:", str(e))
            else:
                text = f"{text.capitalize()}. "
                print(chunk_filename, ":", text)
                whole_text += text
    # get the overall sentiment of the text using VADER
    sentiment_scores = sid.polarity_scores(whole_text)
    sentiment = "Positive" if sentiment_scores["compound"] > 0 else "Negative" if sentiment_scores["compound"] < 0 else "Neutral"
    # return the text and sentiment
    return whole_text, sentiment



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
        else:
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

@app.route('/doc', methods=['POST', 'GET'])
def ind():
    return render_template('home.html')

@app.route('/home', methods=['POST', 'GET'])
def document():
    return render_template('home.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Get the uploaded file from the form
    file = request.files['file']
    
    # Read the contents of the file
    text = file.read().decode('utf-8')
    
    messageBuffer = []
    data = []
    date, time, author = None, None, None
    
    for line in text.split('\n'):
        if date_time(line):
            if len(messageBuffer) > 0:
                data.append([date, time, author, ''.join(messageBuffer)])
                messageBuffer.clear()
            date, time, author, message = get_message(line)
            messageBuffer.append(message)
        else:
            messageBuffer.append(line)
            
    if len(messageBuffer) > 0:
        data.append([date, time, author, ''.join(messageBuffer)])
    
    df = pd.DataFrame(data, columns=["Date", "Time", "Contact", "Message"])
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
   
    
    # Perform sentiment analysis on the entire document
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    sentiment = scores['compound']

    # Classify sentiment as positive, negative or neutral
    if sentiment > 0.05:
        sentiment_class = 'positive'
    elif sentiment < -0.05:
        sentiment_class = 'negative'
    else:
        sentiment_class = 'neutral'
    
# Calculate the number of media files
    media_messages = df[df["Message"]=='<Media omitted>'].shape[0]
    
    num_messages = len(data)
    
    
# Find links in the text and count the matches
    links = re.findall(r'(http[s]?://\S+)', text)
    num_links = len(links)
    





#total emojis 
    total_emojis_list = [a for b in df['Message'] for a in b if a in emoji.UNICODE_EMOJI['en']]
    total_emojis = len(total_emojis_list)
    
# Count the number of occurrences of each emoji
    emoji_counter = Counter()
    for message in df['Message']:
        emojis = [c for c in message if c in emoji.UNICODE_EMOJI['en']]
        emoji_counter.update(emojis)

# Get the most common emoji and its count
    most_common_emoji, count = emoji_counter.most_common(1)[0]
    
# Get the unique contacts from the dataframe
    contacts = df['Contact'].unique().tolist()

# Remove any None values from the list of contacts
    if None in contacts:
        contacts.remove(None)

    # Capitalize each contact name and join them with a newline character
    formatted_contacts = '\n'.join([contact.capitalize() for contact in contacts])

    # Calculate the total number of users
    num_users = len(contacts)
    
    
# convert string dates to datetime format
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

 # start date
    start_date = df['Date'].min().date()

 # end date
    end_date = df['Date'].max().date()

    
    # Define custom color palette
    custom_palette = ['#FFC107', '#03A9F4', '#4CAF50', '#9C27B0', '#F44336', '#673AB7', '#E91E63', '#2196F3', '#FF9800', '#00BCD4']

# Calculate the number of messages for each individual person
    num_messages_per_person = df['Contact'].value_counts().to_dict()
    output_string = ''
    for key, value in num_messages_per_person.items():
        output_string += key + ': ' + str(value) + ', '
    output_string = output_string[:-2] # remove the last comma and space

#most active person
    most_active_person = df['Contact'].value_counts().idxmax()

# Calculate the number of messages for each individual person
    messages_per_person = df['Contact'].value_counts().reset_index()
    messages_per_person.columns = ['contact', 'counts']
    
# Create a bar plot of the number of messages per person
    sns.set(style="whitegrid")
    plt.figure(figsize=(10,6))
    sns.barplot(x='contact', y='counts', data=messages_per_person, palette=custom_palette)
    plt.xticks(rotation=60)
    plt.xlabel('Contact', fontsize=12)
    plt.ylabel('Number of messages', fontsize=12)
    
    plt.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig('static/images/messages_per_person.png')
    
    
# Use Noto Color Emoji font
    plt.rcParams['font.family'] = 'Noto Color Emoji'

    # Define color map for each emoji
    color_map = {
        '😂': 'gold',
        '❤️': 'red',
        '😊': 'orange',
        '😍': 'hotpink',
        '👍': 'limegreen'
    }

    # Find the most common emojis
    emoji_counter = Counter()
    for message in df['Message']:
        emojis = [c for c in message if c in emoji.UNICODE_EMOJI['en']]
        emoji_counter.update(emojis)

    top_emojis = emoji_counter.most_common(5)
    labels = [emoji.emojize(e[0]) for e in top_emojis]
    sizes = [e[1] for e in top_emojis]
    colors = [color_map.get(label, 'gray') for label in labels]

    # Create a pie chart of the top emojis
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, startangle=90, autopct='%1.1f%%', shadow=True)
    ax.axis('equal')
    plt.title('Top 5 Emojis Used')
    plt.tight_layout()
    plt.savefig('static/images/top_emojis.png',dpi=70)
    
    
# Count different emojis used
    emoji_counter = Counter()
    for message in df['Message']:
        emojis = [c for c in message if c in emoji.UNICODE_EMOJI['en']]
        emoji_counter.update(emojis)

# Get the total number of different emojis used
    num_emojis = len(emoji_counter)



    # Define color map for each emoji
    color_map = {
        '😂': 'gold',
        '❤️': 'red',
        '😊': 'orange',
        '😍': 'hotpink',
        '👍': 'limegreen'
    }

    # Count different emojis used
    emoji_counter = Counter()
    for message in df['Message']:
        emojis = [c for c in message if c in emoji.UNICODE_EMOJI['en']]
        emoji_counter.update(emojis)

    # Get the counts for each emoji
    emoji_counts = {emoji: count for emoji, count in emoji_counter.most_common()}

    # Get the top 5 emojis and their counts
    top_emoji_counts = emoji_counter.most_common(5)
    top_emojis = [emoji for emoji, count in top_emoji_counts]
    top_counts = [count for emoji, count in top_emoji_counts]
    other_counts = [count for emoji, count in emoji_counter.items() if emoji not in top_emojis]

    # Define the colors for each emoji
    colors = [color_map.get(label, 'gray') for label in top_emojis] + ['lightgray'] * len(other_counts)

    # Create a bar graph of the top emojis and other emojis
    fig, ax = plt.subplots()
    ax.bar(top_emojis + ['Other'], top_counts + [sum(other_counts)], color=colors)
    ax.set_xticklabels(top_emojis + ['Other'], rotation=0)
    ax.set_ylabel('Count')
    plt.title('Top 5 Emojis Used')
    plt.tight_layout()
    plt.savefig('static/images/top_emojis_bar.png')
    
    
 #week   
# convert the Date column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # extract the day of the week from the Date column
    df['day_of_week'] = df['Date'].dt.day_name()

    # create a bar plot of the number of messages sent by day of the week
    fig, ax = plt.subplots(figsize=(2,1))
    sns.countplot(x='day_of_week', data=df, palette='Set2', ax=ax)

    # highlight the day with the most messages in a different color
    most_active_day = df['day_of_week'].mode().values[0]
    ax.patches[df['day_of_week'].value_counts().index.get_loc(most_active_day)].set_facecolor('crimson')

    # set the labels and title
    ax.set_xlabel('Day of the Week')
    ax.set_ylabel('Number of Messages')
    ax.set_title('Active Day by Messages')

    # save the plot to a file with reduced image size
    fig.savefig('messages_by_day.png', dpi=20)

    
#month    
    # Convert date to month and month name
    df['Month'] = df['Date'].dt.strftime('%Y-%m')
    df['MonthName'] = df['Date'].dt.strftime('%B %Y')

    # Count messages by month
    month_counts = df.groupby('Month')['Message'].count()

    # Define colors for each bar
    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99', '#c2c2f0', '#ffb3e6', '#c2c2f0', '#f0c2c2', '#d9d9d9', '#ffcce6']

    # Plot bar graph with month names and colors
    plt.figure(figsize=(10,8))
    plt.bar(month_counts.index, month_counts.values, color=colors)
    plt.xticks(rotation=45)
    plt.xlabel('Month')
    plt.ylabel('Number of messages')
    

    # Save plot to BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png',dpi=75)
    buffer.seek(0)

    # Embed image in HTML template
    image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    
    
#day    
    # Extract the hour from the Time column
    df['Hour'] = pd.to_datetime(df['Time']).dt.hour

    # Create a new DataFrame with the message counts by hour and contact
    hourly_counts = df.groupby(['Hour', 'Contact'])['Message'].count().reset_index()

    # Use seaborn to create a bar graph
    sns.set(style="whitegrid")
    sns.catplot(x='Hour', y='Message', hue='Contact', data=hourly_counts, kind='bar', height=5.6, aspect=1.3)
    
    # Save the plot to a buffer
    buf = io.BytesIO()

    plt.savefig(buf, format='png',dpi=70)
    buf.seek(0)

    # Encode the plot in base64
    plot_data = base64.b64encode(buf.read()).decode('utf-8')
    
    

    
    # Extract the hour from the Time column
    df['Hour'] = pd.to_datetime(df['Time']).dt.hour

    # Count the number of messages for each hour
    hour_counts = df.groupby('Hour')['Message'].count()

    # Find the hour with the largest message count
    busiest_hour = hour_counts.idxmax()

    # Convert the busiest hour to a string and add "am" or "pm" if necessary
    if busiest_hour < 12:
        busiest_hour_str = str(busiest_hour) + "am"
    else:
        busiest_hour_str = str(busiest_hour-12) + "pm"

    # Get the message count for the busiest hour
    busiest_hour_count = hour_counts[busiest_hour]
    
#find the day with the largest number of messages 
    
    # Extract the date from the Date column
    df['Date'] = pd.to_datetime(df['Date']).dt.date

    # Count the number of messages for each date
    date_counts = df.groupby('Date')['Message'].count()

    # Find the date with the largest message count
    busiest_date = date_counts.idxmax()

    # Convert the busiest date to a string in the desired format
    busiest_date_str = busiest_date.strftime('%Y-%m-%d')

    # Get the message count for the busiest date
    busiest_date_count = date_counts[busiest_date]
    

    
    
    return render_template('result.html',
                                    sentiment=sentiment,
                                    sentiment_class=sentiment_class, 
                                    num_messages=num_messages, 
                                    media_messages=media_messages, 
                                    num_links=num_links, 
                                    total_emojis=total_emojis,
                                    start_date=start_date,
                                    end_date=end_date, 
                                    contacts=formatted_contacts, 
                                    num_messages_per_person=output_string, 
                                    num_users=num_users,
                                    most_common_emoji=most_common_emoji,
                                    count=count,
                                    num_emojis=num_emojis,                                    
                                    most_active_person=most_active_person,
                                    image_path='static/images/messages_per_person.png',
                                    image_path1='static/images/top_emojis.png',
                                    image_path2='static/images/top_emojis_bar.png',
                                    image_file='messages_by_day.png',
                                    image_data=image_data,
                                    plot_data=plot_data,
                                    busiest_hour_str=busiest_hour_str,
                                    busiest_hour_count=busiest_hour_count,
                                    busiest_date_str=busiest_date_str,
                                    busiest_date_count=busiest_date_count)


#sentence

# Define the route for the input page
@app.route('/sen')
def sentence():
    return render_template('sentence.html')

# Define a function for text preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove whitespace
    text = text.strip()
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # POS tag the tokens
    pos_tokens = nltk.pos_tag(tokens)
    # Join the tokens back into a string
    text = " ".join(tokens)
    return text

# Define the behaviour labels
Behaviour_labels = {
    "Crime":["i will kill you","kill","killed","kill me","killing","i want to kill you"],
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


# Define a function to map the predicted label to the corresponding emotion label
def map_Behaviour_label(label):
    for Behaviour, keywords in Behaviour_labels.items():
        if any(keyword in label.lower() for keyword in keywords):
            return Behaviour
    return None

# Define function to get sentiment label
def get_sentiment_label(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    if sentiment['compound'] >= 0:
        return "positive"
    else:
        return "negative"

# Define the pipeline for sentiment analysis
tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
model = TFRobertaForSequenceClassification.from_pretrained("arpanghoshal/EmoRoBERTa")
Behaviour = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# Define the vectorizer for SVM
vectorizer = TfidfVectorizer()

# Load the data
df = pd.read_csv("behaviour.csv")

# Apply the text preprocessing function to the text data
df["Message"] = df["Message"].apply(preprocess_text)

# Split the data into train and test sets for SVM
X_train, X_test, y_train, y_test = train_test_split(df['Message'], df['Behaviour'], test_size=0.2, random_state=42)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train the SVM model
svm = LinearSVC()
svm.fit(X_train, y_train)

def predict_Behaviour_label(text):
    result = Behaviour(text)[0]
    Behaviour_label = map_Behaviour_label(result["label"])
    if Behaviour_label is None:
        text_vectorized = vectorizer.transform([text])
        Behaviour_label = svm.predict(text_vectorized)[0]
    return Behaviour_label


@app.route("/pre", methods=["POST","GET"])
def senpre():
    text = request.form['text']
    Behaviour_label = predict_Behaviour_label(text)
    sentiment_label = get_sentiment_label(text)
    output = ""

    if Behaviour_label == "Crime":
        if sentiment_label == "positive":
            output = "The sentiment in the sentence is classified as positive and it is labeled as Non-voilent."
        else:
            output = "The sentiment in the sentence is classified as negative and it is labeled as Crime."
            
    elif Behaviour_label == "Short-temper":
        if sentiment_label == "positive":
            output = "The sentiment in the sentence is classified as positive and it is labeled as Calm"
        else:
            output = "The sentiment in the sentence is classified as negative and it is labeled as short-temper."
            
    elif Behaviour_label == "Appreciation":
        if sentiment_label == "positive":
            output = "The sentiment in the sentence is classified as positive and it is labeled as Appreciation."
        else:
            output = "The sentiment in the sentence is classified as negative and it is labeled as Depreciation"
            
    elif Behaviour_label == "Enthusiasm":
        if sentiment_label == "positive":
            output = "The Sentiment in the sentence is classified as positive and it labeled as Enthusiasm"
        else:
            output = "The Sentiment in the sentence is classified as negative and it labeled as Unenthusiasm"
    
    elif Behaviour_label == "Optimistic":
        if sentiment_label == "positive":
            output = "The Sentiment in the sentence is classified as positive and it labeled as Optimistic"
        else:
            output = "The Sentiment in the sentence is classified as negative and it labeled as Pesimestic"

    elif Behaviour_label == "Caring":
        if sentiment_label == "positive":
            output = "The Sentiment in the sentence is classified as positive and it labeled as Caring"
        else:
            output = "The Sentiment in the sentence is classified as negative and it labeled as Careless"

    elif Behaviour_label == "Relief":
        if sentiment_label == "positive":
            output = "The Sentiment in the sentence is classified as positive and it labeled as Relief"
        else:
            output = "The Sentiment in the sentence is classified as negative and it labeled as Discomfort"

    elif Behaviour_label == "Depressed":
        if sentiment_label == "positive":
            output = "The Sentiment in the sentence is classified as positive and it labeled as Happy"
        else:
            output = "The Sentiment in the sentence is classified as negative and it labeled as Depressed"

    elif Behaviour_label == "Disgust":
        if sentiment_label == "positive":
            output = "The Sentiment in the sentence is classified as positive and it labeled as Attraction"
        else:
            output = "The Sentiment in the sentence is classified as negative and it labeled as  Disgust"


    elif Behaviour_label == "Introvert":
        if sentiment_label == "positive":
           output = "The Sentiment in the sentence is classified as positive and it labeled as Extrovert"
        else:
            output = "The Sentiment in the sentence is classified as negative and it labeled as  introver"


    elif Behaviour_label == "Clever":
        if sentiment_label == "positive":
            output = "The Sentiment in the sentence is classified as positive and it labeled as Clever"
        else:
            output = "The Sentiment in the sentence is classified as negative and it labeled as Foolish"

    elif Behaviour_label == "Friendly":
        if sentiment_label == "positive":
            output = "The Sentiment in the sentence is classified as positive and it labeled as Friendly"
        else:
            output = "The Sentiment in the sentence is classified as negative and it labeled as Unfriendly"

    elif Behaviour_label == "Helpful":
        if sentiment_label == "positive":
            output = "The Sentiment in the sentence is classified as positive and it labeled as Helpful"
        else:
            output = "The Sentiment in the sentence is classified as negative and it labeled as Unhelpful"

    elif Behaviour_label == "mental illness":
        if sentiment_label == "positive":
            output = "The Sentiment in the sentence is classified as postive and it labeled as Stable"
        else:
            output = "The Sentiment in the sentence is classified as negative and it labeled as mental illness"

    elif Behaviour_label == "Confident":
        if sentiment_label == "positive":
            output = "The Sentiment in the sentence is classified as positive and it labeled as Confident"
        else:
            output = "The Sentiment in the sentence is classified as negative and it labeled as Unconfident"

    elif Behaviour_label == "selfish":
        if sentiment_label == "positive":
            output = "The Sentiment in the sentence is classified as positive and it labeled as Unselfish"
        else:
            output = "The Sentiment in the sentence is classified as negative and it labeled as selfish"

    elif Behaviour_label == "Honest":
        if sentiment_label == "positive":
            output = "The Sentiment in the sentence is classified as positive and it labeled as Honest"
        else:
            output = "The Sentiment in the sentence is classified as negative and it labeled as Dishonest"


    elif Behaviour_label == "laziness ":
        if sentiment_label == "positive":
            output = "The Sentiment in the sentence is classified as positive and it labeled as Active"
        else:
            output = "The Sentiment in the sentence is classified as negative and it labeled as  laziness"

    elif Behaviour_label == "Relaxation":
        if sentiment_label == "positive":
            output = "The Sentiment in the sentence is classified as positive and it labeled as Relaxation "
        else:
            output = "The Sentiment in the sentence is classified as negative and it labeled as  Discomfort"


    # Render the result page with the predicted label
    return render_template("pre.html",output=output)



# Define the route for the input page
@app.route('/voice')
def voice():
    return render_template('voice_input.html')
	

#speech to text
def convert_speech_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak anything...")
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        print("You said: {}".format(text))
        return text
    except:
        print("Sorry, could not recognize your voice")
        return None


# Define a function for text preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove whitespace
    text = text.strip()
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # POS tag the tokens
    pos_tokens = nltk.pos_tag(tokens)
    # Join the tokens back into a string
    text = " ".join(tokens)
    return text

# Define the behaviour labels
Behaviour_labels = {
     "Crime":["i will kill you","kill","killed","kill me","killing","i want to kill you"],
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


# Define a function to map the predicted label to the corresponding emotion label
def map_Behaviour_label(label):
    for Behaviour, keywords in Behaviour_labels.items():
        if any(keyword in label.lower() for keyword in keywords):
            return Behaviour
    return None

# Define function to get sentiment label
def get_sentiment_label(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    if sentiment['compound'] >= 0:
        return "positive"
    else:
        return "negative"

# Define the pipeline for sentiment analysis
tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
model = TFRobertaForSequenceClassification.from_pretrained("arpanghoshal/EmoRoBERTa")
Behaviour = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# Define the vectorizer for SVM
vectorizer = TfidfVectorizer()

# Load the data
df = pd.read_csv("behaviour.csv")

# Apply the text preprocessing function to the text data
df["Message"] = df["Message"].apply(preprocess_text)

# Split the data into train and test sets for SVM
X_train, X_test, y_train, y_test = train_test_split(df['Message'], df['Behaviour'], test_size=0.2, random_state=42)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train the SVM model
svm = LinearSVC()
svm.fit(X_train, y_train)

def predict_Behaviour_label(text):
    result = Behaviour(text)[0]
    Behaviour_label = map_Behaviour_label(result["label"])
    if Behaviour_label is None:
        text_vectorized = vectorizer.transform([text])
        Behaviour_label = svm.predict(text_vectorized)[0]
    return Behaviour_label


@app.route("/voicepre", methods=["POST","GET"])
def voicepre():
    text = request.form['text']
    if not text:
        text = convert_speech_to_text()
        if not text:
            return render_template('voice_input.html', prediction='Sorry, could not recognize your voice')
    Behaviour_label = predict_Behaviour_label(text)
    sentiment_label = get_sentiment_label(text)
    output = ""

    if Behaviour_label == "Crime":
        if sentiment_label == "positive":
            output = "The sentiment in the sentence is classified as positive and it is labeled as Non-voilent."
        else:
            output = "The sentiment in the sentence is classified as negative and it is labeled as Crime."
            
    elif Behaviour_label == "short-temper":
        if sentiment_label == "positive":
            output = "The sentiment in the sentence is classified as positive and it is labeled as Calm"
        else:
            output = "The sentiment in the sentence is classified as negative and it is labeled as short-temper."
            
    elif Behaviour_label == "Appreciation":
        if sentiment_label == "positive":
            output = "The sentiment in the sentence is classified as positive and it is labeled as Appreciation."
        else:
            output = "The sentiment in the sentence is classified as negative and it is labeled as Depreciation"
            
    elif Behaviour_label == "Enthusiasm":
        if sentiment_label == "positive":
            output = "The Sentiment in the sentence is classified as positive and it labeled as Enthusiasm"
        else:
            output = "The Sentiment in the sentence is classified as negative and it labeled as Unenthusiasm"
    
    elif Behaviour_label == "Optimistic":
        if sentiment_label == "positive":
            output = "The Sentiment in the sentence is classified as positive and it labeled as Optimistic"
        else:
            output = "The Sentiment in the sentence is classified as negative and it labeled as Pesimestic"

    elif Behaviour_label == "Caring":
        if sentiment_label == "positive":
            output = "The Sentiment in the sentence is classified as positive and it labeled as Caring"
        else:
            output = "The Sentiment in the sentence is classified as negative and it labeled as Careless"

    elif Behaviour_label == "Relief":
        if sentiment_label == "positive":
            output = "The Sentiment in the sentence is classified as positive and it labeled as Relief"
        else:
            output = "The Sentiment in the sentence is classified as negative and it labeled as Discomfort"

    elif Behaviour_label == "Depressed":
        if sentiment_label == "positive":
            output = "The Sentiment in the sentence is classified as positive and it labeled as Happy"
        else:
            output = "The Sentiment in the sentence is classified as negative and it labeled as Depressed"

    elif Behaviour_label == "Disgust":
        if sentiment_label == "positive":
            output = "The Sentiment in the sentence is classified as positive and it labeled as Attraction"
        else:
            output = "The Sentiment in the sentence is classified as negative and it labeled as  Disgust"


    elif Behaviour_label == "introvert":
        if sentiment_label == "positive":
           output = "The Sentiment in the sentence is classified as positive and it labeled as Extrovert"
        else:
            output = "The Sentiment in the sentence is classified as negative and it labeled as  introver"


    elif Behaviour_label == "Clever":
        if sentiment_label == "positive":
            output = "The Sentiment in the sentence is classified as positive and it labeled as Clever"
        else:
            output = "The Sentiment in the sentence is classified as negative and it labeled as Foolish"

    elif Behaviour_label == "Friendly":
        if sentiment_label == "positive":
            output = "The Sentiment in the sentence is classified as positive and it labeled as Friendly"
        else:
            output = "The Sentiment in the sentence is classified as negative and it labeled as Unfriendly"

    elif Behaviour_label == "Helpful":
         if sentiment_label == "positive":
            output = "The Sentiment in the sentence is classified as positive and it labeled as Helpful"
         else:
            output = "The Sentiment in the sentence is classified as negative and it labeled as Unhelpful"

    elif Behaviour_label == "mental illness":
        if sentiment_label == "positive":
            output = "The Sentiment in the sentence is classified as postive and it labeled as Stable"
        else:
            output = "The Sentiment in the sentence is classified as negative and it labeled as mental illness"

    elif Behaviour_label == "Confident":
        if sentiment_label == "positive":
            output = "The Sentiment in the sentence is classified as positive and it labeled as Confident"
        else:
            output = "The Sentiment in the sentence is classified as negative and it labeled as Unconfident"

    elif Behaviour_label == "selfish":
        if sentiment_label == "positive":
            output = "The Sentiment in the sentence is classified as positive and it labeled as Unselfish"
        else:
            output = "The Sentiment in the sentence is classified as negative and it labeled as selfish"

    elif Behaviour_label == "Honest":
        if sentiment_label == "positive":
            output = "The Sentiment in the sentence is classified as positive and it labeled as Honest"
        else:
            output = "The Sentiment in the sentence is classified as negative and it labeled as Dishonest"


    elif Behaviour_label == "laziness ":
        if sentiment_label == "positive":
            output = "The Sentiment in the sentence is classified as positive and it labeled as Active"
        else:
            output = "The Sentiment in the sentence is classified as negative and it labeled as  laziness"

    elif Behaviour_label == "Relaxation":
        if sentiment_label == "positive":
            output = "The Sentiment in the sentence is classified as positive and it labeled as Relaxation "
        else:
            output = "The Sentiment in the sentence is classified as negative and it labeled as  Discomfort"


    # Render the result page with the predicted label
    return render_template("voicepre.html",output=output)


@app.route('/audio', methods=['GET', 'POST'])
def audio():
    if request.method == 'POST':
        # get the uploaded file
        file = request.files['file']
        # save the file
        file.save('audio.wav')
        # get the transcribed text and sentiment
        text, sentiment = get_large_audio_transcription('audio.wav')
        # delete the audio file
        os.remove('audio.wav')
        # render the template with the transcribed text and sentiment
        return render_template('audio.html', text=text, sentiment=sentiment)
    # render the template with the form
    return render_template('audio.html')

                           
@app.route('/para')
def para():
    return render_template('para.html')                          

@app.route('/paragraph_sentiment', methods=['POST'])
def paragraph_sentiment():
    paragraph = request.form['paragraph']
    label, translated_paragraph = get_paragraph_sentiment(paragraph)
    return render_template('paragraph_sentiment.html', paragraph=paragraph, label=label, translated_paragraph=translated_paragraph)


        
if __name__ == '__main__':
    app.run(debug=True)
