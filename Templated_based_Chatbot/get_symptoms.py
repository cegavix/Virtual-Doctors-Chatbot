import string

import numpy as np
import pandas as pd
import nltk
from joblib import load
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from util import make_arrays_from_csv

from simple_colors import *

# TODO: Change the data, take out high risk diagnosis, like AIDS.
def extract_keywords(text):
    """
    Generates ngrams of KEYWORDS in the format of the column titles, free of punctuation and stop words.
        :param text (str): The text to preprocess.
        :return: tokens, A list of preprocessed tokens.
    """
    # Custom stop words built upon gensim.stopwords
    stop_words = (
        {'want', 'diagnose', 'diagnosis', 'sick', 'illness', 'i', 'those', 'own', 'yourselves', 'ie', 'around',
         'between', 'four', 'been', 'alone', 'off', 'am', 'then', 'other', 'can', 'cry', 'regarding', 'hereafter',
         'front',
         'too', 'used', 'wherein', 'doing', 'everything', 'up', 'never', 'onto', 'how', 'either', 'before', 'anyway',
         'since', 'through', 'amount', 'now', 'he', 'cant', 'was', 'con', 'have', 'into', 'because', 'inc', 'not',
         'therefore', 'they', 'even', 'whom', 'it', 'see', 'somewhere', 'interest', 'thereupon', 'thick', 'nothing',
         'whereas', 'much', 'whenever', 'find', 'seem', 'until', 'whereby', 'at', 'ltd', 'fire', 'also', 'some', 'last',
         'than', 'get', 'already', 'our', 'doesn', 'once', 'will', 'noone', 'that', 'what', 'thus', 'no', 'myself',
         'out',
         'next', 'whatever', 'although', 'though', 'etc', 'which', 'would', 'therein', 'nor', 'somehow', 'whereupon',
         'besides', 'whoever', 'thin', 'ourselves', 'few', 'did', 'third', 'without', 'twelve', 'anything', 'against',
         'while', 'twenty', 'if', 'however', 'found', 'herself', 'when', 'may', 'six', 'ours', 'done', 'seems', 'else',
         'call', 'perhaps', 'had', 'nevertheless', 'fill', 'where', 'otherwise', 'still', 'within', 'its', 'for',
         'together', 'elsewhere', 'throughout', 'eg', 'others', 'show', 'sincere', 'anywhere', 'anyhow', 'as', 'are',
         'the', 'hence', 'something', 'hereby', 'nowhere', 'latterly', 'de', 'say', 'does', 'neither', 'his', 'go',
         'forty',
         'put', 'their', 'by', 'namely', 'km', 'could', 'five', 'unless', 'itself', 'is', 'nine', 'whereafter', 'down',
         'bottom', 'thereby', 'such', 'both', 'she', 'become', 'whole', 'who', 'yourself', 'every', 'thru', 'except',
         'very', 'several', 'among', 'being', 'be', 'mine', 'further', 'here', 'during', 'why', 'with', 'just',
         'becomes',
         'about', 'a', 'co', 'using', 'seeming', 'due', 'wherever', 'beforehand', 'detail', 'fifty', 'becoming',
         'might',
         'amongst', 'my', 'empty', 'thence', 'thereafter', 'almost', 'least', 'someone', 'often', 'from', 'keep', 'him',
         'or', 'top', 'her', 'didn', 'nobody', 'sometime', 'across', 'hundred', 'only', 'via', 'name', 'eight', 'three',
         'back', 'to', 'all', 'became', 'move', 'me', 'we', 'formerly', 'so', 'i', 'whence', 'describe', 'under',
         'always',
         'himself', 'more', 'herein', 'in', 'after', 'themselves', 'you', 'them', 'above', 'sixty', 'hasnt', 'your',
         'made',
         'everywhere', 'indeed', 'most', 'kg', 'fifteen', 'but', 'must', 'along', 'beside', 'hers', 'computer', 'side',
         'former', 'full', 'anyone', 'has', 'yours', 'whose', 'behind', 'please', 'mill', 'amoungst', 'ten', 'seemed',
         'sometimes', 'should', 'over', 'take', 'each', 'don', 'same', 'rather', 'really', 'latter', 'and', 'part',
         'hereupon', 'per', 'eleven', 'ever', 'enough', 'again', 'us', 'yet', 'moreover', 'mostly', 'one', 'meanwhile',
         'whither', 'there', 'toward', 'give', 'system', 'do', 'quite', 'an', 'these', 'everyone', 'towards', 'this',
         'bill', 'cannot', 'un', 'afterwards', 'beyond', 'make', 'were', 'whether', 'well', 'another', 'below', 'first',
         'upon', 'any', 'none', 'many', 'various', 'serious', 're', 'two', 'less', 'couldnt'})
    # Tokenize the text
    tokens = nltk.word_tokenize(text.lower())
    # Remove punctuation
    tokens = [word for word in tokens if word not in string.punctuation]
    tokens = [word for word in tokens if word not in stop_words]
    ngrams_list = []
    for n in range(1, 4):  # Generate 1-grams, 2-grams, and 3-grams
        ngrams1 = ngrams(tokens, n)
        ngrams_list.extend(["_".join(gram) for gram in ngrams1])

    keywords = " ".join(ngrams_list)
    return keywords


def ask_symptoms(user_input, found):
    """
    Recursive function to ask for symptoms that builds upon found every recursion
    :param user_keywords: the preprocessed user input
    :param found: the found symptoms, starts empty and it built upon in every recursion
    :return: dataframe with symptoms for the ml model
    """
    if (user_input.lower() == 'no'):
        # Symptoms Collected. Construct input for the ML model
        df = pd.DataFrame(columns=all_symptoms_list)
        # One-hot encode using loc for found symptom columns
        df.loc[0] = 0
        df.loc[0, found] = 1
        return df

    # Extract symptoms from user input into 1 word format:
    # ie acute_kidney_failure rather than acute kidney failure
    user_keywords = extract_keywords(user_input)

    # Matching symptoms
    for symptom in all_symptoms_list:
        if symptom.lower() not in found:
            if symptom.lower() in user_keywords.lower():
                found.append(symptom)

    if found:
        # TODO: Print the symptoms pretty, not in array format. Dont repeat symptoms (this might be hard w recursion)
        user_input = input(blue(
            "Neutrino: I see you are experiencing %s. Do you have any other symptoms? List them, or say no." % found))
    else:
        user_input = input(blue("Neutrino: What are your symptoms?"))

    #  Recursive call to ask for more symptoms
    df = ask_symptoms(user_input, found)
    return df


def predict_with_confidence(found_symptoms_df, attempted):
    # Use the model to predict the illness with a measure of confidence
    fitted_svm_model = load('saved_structures/xgboost.joblib')

    svm_proba_val = fitted_svm_model.predict_proba(found_symptoms_df)
    # print("SVM probabilities:", svm_proba_val[0])
    # Get the max probability, or the next max if already tried
    # sorted_indices = np.argsort(svm_proba_val[0])
    # current_max = sorted_indices[-attempts]
    # max_proba = svm_proba_val[0][current_max]
    max_proba = np.max(svm_proba_val[0])

    # # Get the prediction out of the ndnumpy format -> string format
    pred_key = (fitted_svm_model.predict(found_symptoms_df)).item(0)

    disease, key = make_arrays_from_csv('Datasets/util/disease_key.csv')

    index = key.index(str(pred_key))
    pred_name = disease[index]

    # print('Attempt num:', attempted, 'The prediction we re checking is:', pred_name)

    if attempted == True and max_proba > 0.8:
        print(blue(
            "Neutrino: I think you might have %s. Would you like any more information or treatment advice? I recommend getting a GP appointment. If the diagnosis doesn't sound right, feel free to give me some more symptoms." % pred_name))
        # TODO: Make a dictionary of disease, symptoms, and treatments options and advice for users to search through
        return pred_name, 'CONFIDENT'
    elif attempted == False and max_proba >0.1:
        print(blue(
            "Neutrino: I suspect you might have %s. However, I'm going to need to ask you some follow up questions." % pred_name))
        found_symptoms_df = further_symptoms_questions(pred_name, found_symptoms_df)
        pred_name, confidence = predict_with_confidence(found_symptoms_df, True)
        if confidence == 'CONFIDENT':
            return pred_name, 'CONFIDENT'  # only returns if confident on 2nd try
    else:
        print(blue("Neutrino: I am afraid your symptoms are too general for me to confidentally diagnose you. If you have "
              "multiple potential illnesses, try again and describe one set of symptoms at a time. Otherwise, "
              "please make an appointment to verify any advice or diagnoses by a  verified GP."))
        return pred_name, 'NOT FOUND'
    return pred_name, 'NOT FOUND'


def further_symptoms_questions(pred_name, found_symptoms_df):
    found_symptoms_names = found_symptoms_df.loc[:, (found_symptoms_df == 1).any()].columns.tolist()

    # 1. Get the row of the true diagnosis
    true_row = training_df[training_df['prognosis'] == pred_name]

    # 2. Get the symptoms names of the true row, where the value is 1
    true_row = true_row.drop(columns=['prognosis'])
    true_symptoms = true_row.loc[:, (true_row == 1).any()].columns.tolist()
    # print("True symptoms this disease : ", true_symptoms)
    # print("Your symptoms: ", found_symptoms_names)

    for symptom in true_symptoms:
        if symptom not in found_symptoms_names:
            pretty_symptom = symptom.replace('_', ' ')
            user_input = input("Neutrino: Do you have %s?" % pretty_symptom)
            if user_input.lower() == 'yes':
                found_symptoms_names.append(symptom)

    # Symptoms Collected. Construct input for the ML model
    df = pd.DataFrame(columns=all_symptoms_list)
    # One-hot encode using loc for found symptom columns
    df.loc[0] = 0
    df.loc[0, found_symptoms_names] = 1
    return df


training_df = pd.read_csv('Datasets/custom_testing.csv')
all_symptoms_list = training_df.columns.tolist()[5:]
disease = training_df['prognosis'].values
information = training_df['definition'].values
treatments = training_df['Treatment'].values
urgent_appointment = training_df['urgent_appointment'].values

# Drop the columns that are not needed for the model. Only need x and y.
training_df = training_df.drop(columns=['definition', 'Treatment', 'urgent_appointment'])
