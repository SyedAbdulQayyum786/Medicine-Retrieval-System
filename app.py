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
        # Get user input from the form
        symptoms = request.form['symptoms']
        
        
        # Preprocess the symptoms
        preprocessed_symptoms = preprocess_text(symptoms)
        print(preprocessed_symptoms)
        
        # Split the preprocessed symptoms by comma with and without spaces
        symptoms_list = [symptom.strip() for symptom in preprocessed_symptoms.split(',')]
        print(symptoms_list)
        
        # Read the CSV file into a pandas DataFrame
        file_path = "Medicine_Details.csv" 
        df = pd.read_csv(file_path, usecols=["Medicine Name", "Uses"])
        
        # Preprocess the DataFrame
        df = preprocess_dataframe(df)
        
        # Filter DataFrame based on symptoms using AND condition
        result_df = df.copy()
        for symptom in symptoms_list:
            result_df = result_df[result_df['Uses'].str.contains(symptom, case=False)]
        
        # Calculate score for each medicine based on number of symptoms matched
        result_df['Score'] = result_df['Uses'].apply(lambda x: sum(symptom in x.lower() for symptom in symptoms_list))
        
        # Sort the DataFrame based on score in descending order
        result_df.sort_values(by=['Score', 'Medicine Name'], ascending=[False, True], inplace=True)
        
        # Retrieve top-ranked medicines
        top_medicines = result_df.head(10)  # Retrieve top 3 medicines
        
        # If no medicine found, retrieve 3 medicines with individual symptoms
        if top_medicines.empty:
            # Reset the filter to include all medicines
            result_df = df.copy()
            
            # Filter DataFrame based on individual symptoms using OR condition
            for symptom in symptoms_list:
                result_df = result_df[result_df['Uses'].str.contains(symptom, case=False)]
            
            # Calculate score for each medicine based on number of symptoms matched
            result_df['Score'] = result_df['Uses'].apply(lambda x: sum(symptom in x.lower() for symptom in symptoms_list))
            
            # Sort the DataFrame based on score in descending order
            result_df.sort_values(by=['Score', 'Medicine Name'], ascending=[False, True], inplace=True)
            
            # Retrieve top-ranked medicines with individual symptoms
            top_medicines = result_df.head(3)  # Retrieve top 3 medicines
        
        return render_template('results.html', result=top_medicines.to_html())
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
