import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords

def extract_symptoms(text):
    """
    Generates ngrams of KEYWORDS in the format of the column titles, free of punctuation and stop words.
        :param text (str): The text to preprocess.
        :return: tokens, A list of preprocessed tokens.
    """
    # Tokenize the text
    tokens = nltk.word_tokenize(text.lower())
    # Remove punctuation
    tokens = [word for word in tokens if word not in string.punctuation]
    tokens = [word for word in tokens if word not in stop_words]
    ngrams_list = []
    for n in range(1, 4):  # Generate 1-grams, 2-grams, and 3-grams
        ngrams1 = ngrams(tokens, n)
        ngrams_list.extend(["_".join(gram) for gram in ngrams1])

    return ngrams_list

#WORKING ONE
def ask_symptoms(user_keywords, found):
    """
    Recursive function to ask for symptoms that builds upon found every recursion
    :param user_keywords: the preprocessed user input
    :param found: the found symptoms, starts empty and it built upon in every recursion
    :return: dataframe with symptoms for the ml model
    """
    if (user_keywords.lower() == 'no'):
        # Symptoms Collected. Construct input for the ML model
        df = pd.DataFrame(columns=symptoms_list)
        # One-hot encode using loc for found symptom columns
        df.loc[0] = 0
        df.loc[0, found] = 1
        print("Your symptoms dataframe:")
        print(df)
        return df

    # Extract symptoms from user input into 1 word format: ie acute_kidney_failure rather than kidney failure
    user_keywords = extract_symptoms(user_keywords)
    # Look for matching symptoms
    for symptom in symptoms_list:
        if symptom.lower() not in found:
            if symptom.lower() in user_keywords.lower():
                print("Matched: ", symptom)
                found.append(symptom)

    if found:
        user_input = input("Neutrino: I see you are experiencing %s. Do you have any other symptoms? If this is all, please say no." % found)
    else:
        user_input = input("Neutrino: What are your symptoms?")

    #  Recursive call to ask for more symptoms
    return ask_symptoms(user_input, found)






symptoms = pd.read_csv('Datasets/KAUSHIL268-dataset/Training.csv')
symptoms_list = symptoms.columns.tolist()[:-2]

df = ask_symptoms("I want you to diagnose my illness", [])