from flask import Flask, render_template, request
import pandas as pd
from spellchecker import SpellChecker
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

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
    
    words = text.split()

    
    misspelled = spell.unknown(words)

   
    corrected_text = []
    for word in words:
        corrected_text.append(spell.correction(word) if word in misspelled else word)

    return ' '.join(corrected_text)

def preprocess_text(text):
    try:
        new_corrected = spell_check(text)
        
       
        tokens = nltk.word_tokenize(new_corrected)
        
        
        filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
        
        
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
        
      
        stemmed_tokens = [stemmer.stem(token) for token in lemmatized_tokens]
        
        return ' '.join(stemmed_tokens)
    except Exception as e:
        print("Error occurred during text preprocessing:", e)
        return text  


# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
       
        symptoms = request.form['symptoms']
        
       
        preprocessed_symptoms = preprocess_text(symptoms)

        
        file_path = "Medicine_Details.csv" 
        df = pd.read_csv(file_path, usecols=["Medicine Name", "Uses"])

       
        df = preprocess_dataframe(df)

     
        result_df = df[df['Uses'].str.contains(preprocessed_symptoms, case=False)]

        return render_template('results.html', result=result_df.to_html())
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
