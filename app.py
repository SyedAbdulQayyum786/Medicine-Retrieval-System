from flask import Flask, render_template, request
import pandas as pd
from spellchecker import SpellChecker
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import re
app = Flask(__name__)

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
spell = SpellChecker()

def preprocess_dataframe(df):
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df

def spell_check(text):
    words = re.split(r',|\s', text)
    # Filter out empty strings and strip whitespace from each word
    words = [word.strip() for word in words if word.strip()]
    corrected_text = []
    for word in words:
        corrected_word = spell.correction(word)
        if corrected_word is None or corrected_word == word:
            corrected_text.append(word)
        else:
            corrected_text.append(corrected_word)
    return ' '.join(corrected_text)

def preprocess_text(text):
    try:
        new_corrected = spell_check(text)
        if new_corrected.strip() == '':
            return text
        tokens = nltk.word_tokenize(new_corrected)
        filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
        stemmed_tokens = [stemmer.stem(token) for token in lemmatized_tokens]
        return ' '.join(stemmed_tokens)
    except Exception as e:
        print("Error occurred during text preprocessing:", e)
        return text

def search_medicines(symptoms_list, df):
    result_df = df.copy()
    for symptom in symptoms_list:
        result_df = result_df[result_df['Uses'].str.contains(symptom, case=False)]
    result_df['Score'] = result_df['Uses'].apply(lambda x: sum(symptom in x.lower() for symptom in symptoms_list))
    result_df.sort_values(by=['Score', 'Medicine Name'], ascending=[False, True], inplace=True)
    return result_df[['Medicine Name', 'Uses']].head(10)

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        symptoms = request.form['symptoms']
        preprocessed_symptoms = preprocess_text(symptoms)
        symptoms_list = [symptom.strip() for symptom in preprocessed_symptoms.split(' ')]
        file_path = "Medicine_Details.csv" 
        df = pd.read_csv(file_path, usecols=["Medicine Name", "Uses"])
        df = preprocess_dataframe(df)
        result_df = df.copy()
        for symptom in symptoms_list:
            result_df = result_df[result_df['Uses'].str.contains(symptom, case=False)]
        result_df['Score'] = result_df['Uses'].apply(lambda x: sum(symptom in x.lower() for symptom in symptoms_list))
        result_df.sort_values(by=['Score', 'Medicine Name'], ascending=[False, True], inplace=True)
        top_medicines = result_df.head(10)  
        if top_medicines.empty:
            result_df = df.copy()
            individual_medicines = {}
            for symptom in symptoms_list:
                individual_result = df[df['Uses'].str.contains(symptom, case=False)]
                individual_medicines[symptom] = individual_result.head(10)
            return render_template('individual_results.html', results=individual_medicines)

        return render_template('results.html', result=result_df.to_html())

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
