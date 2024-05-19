import joblib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, multilabel_confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from simple_colors import *
from util import make_arrays_from_csv

from get_symptoms import treatments, disease, information, urgent_appointment

def find_similar_q(my_question, name):
    """
    :param my_question: the user input that will be assessed to see if it matches an intent
    :return: bool return as to whether a intent was found or not.

    This is a slow function, hence it coming at the end of the intent matching hierarchy.
    """

    make_vector_space_with_Classifier()
    # Maps onto the same vector space AND tokenizes and tfidf weighs them
    questions, answers = make_arrays_from_csv('datasets/QA_dataset.csv')
    tfidf_vectorizer = joblib.load('qa_vectorizer.joblib')
    tfidf_matrix = tfidf_vectorizer.transform(questions)

    user_vector = tfidf_vectorizer.transform([my_question])
    # Calculate cosine similarity between the search vector and all question vectors
    cosine_similarities = cosine_similarity(user_vector, tfidf_matrix)

    cosine_similarities = cosine_similarities.flatten()
    most_similar_indexes = cosine_similarities.argsort()[::-1][:3]  # -1 starts from the end, 3 returns array of top
    # Get the most similar question, only add to the array if the values are above 0.6

    index_of_possible_matches = []
    for index in most_similar_indexes:
        if cosine_similarities[index] > 0.5:
            index_of_possible_matches.append(index)

    count = len(index_of_possible_matches)

    if count == 0:
        # No matches found.
        return False
    elif my_question == questions[index_of_possible_matches[0]] or count == 1:
        print(answers[most_similar_indexes[0]])
        return True

def make_vector_space_with_Classifier():
    """
    Makes the vectorizers for smalltalk and intents in joblibs since they are unchanging. Makes the classifier for
    intents.
    """
    # Only needs to be run once (unless datasets are changed). Stop words make the questions useless???
    st_vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    intent_tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), lowercase=True)

    # Make Classifier
    responses, intents = make_arrays_from_csv('Datasets/intentmatch_dataset.csv')
    print(intents)
    intent_tfidf_matrix = intent_tfidf_vectorizer.fit_transform(responses)
    classifier = DecisionTreeClassifier(random_state=0).fit(intent_tfidf_matrix, intents)

    # Dump data structures
    joblib.dump(intent_tfidf_matrix, 'saved_structures/intent_matrix.joblib')
    joblib.dump(st_vectorizer, 'saved_structures/st_vectorizer.joblib')
    joblib.dump(intent_tfidf_vectorizer, 'saved_structures/intent_vectorizer.joblib')
    joblib.dump(classifier, 'saved_structures/intent_classifier.joblib')

# make_vector_space_with_Classifier()
def small_talk_similarity(user_response_of_intent, current_highest_similarity):
    """
    Calculates the cosine similarity between the user input and the small talk dataset. If the similarity is higher than
    0.6 it is matched

    :param user_response_of_intent:
    :param current_highest_similarity:
    :return: best_st_response: returns the best small talk response if there is one
    """

    st_vectorizer = joblib.load('saved_structures/st_vectorizer.joblib')
    # Make term freq matrices for small talk
    st_prompt, st_responses = make_arrays_from_csv('Datasets/smalltalk_dataset.csv')
    st_matrix = st_vectorizer.fit_transform(st_prompt)

    st_user_vector = st_vectorizer.transform([user_response_of_intent])
    st_cosine_similarities = cosine_similarity(st_user_vector, st_matrix)

    # Get the highest cosine similarity for small talk
    st_most_similar_index = st_cosine_similarities.argmax()
    st_highest_similarity = st_cosine_similarities[0, st_most_similar_index]

    if st_highest_similarity > current_highest_similarity and st_highest_similarity > 0.6:
        best_st_response = st_responses[st_most_similar_index]
        return best_st_response
    else: return None



def classify_intent_similarity(user_response_of_intent, name):
    """
    Handles the dynamic aspects of vectorizers, that rely on user input
    Uses both Cosine Similarity and a Classifier to determine sure intent. SMALL TALK is a separate Vector Space and
    uses only cosine similarity. Other intents are in a Vector space together and use a Classifier AND Cosine Similarity.

    :param user_response_of_intent: str, what the user inputted
    :return: Provides the intent, which can be any of the following: small talk, name, booking, menu, exit
    """
    # Load in vectorizers
    intent_vectorizer = joblib.load('saved_structures/intent_vectorizer.joblib')

    # Map user input onto small talk and intent vectorizers
    user_vector = intent_vectorizer.transform([user_response_of_intent])

    # Predict the intent using classifier
    classifier = joblib.load('saved_structures/intent_classifier.joblib')
    pred_intent = classifier.predict(user_vector)

    # Calculate cosine similarity between the input vector and intent vectors and small talk
    tfidf_matrix = joblib.load('saved_structures/intent_matrix.joblib')
    cosine_similarities = cosine_similarity(user_vector, tfidf_matrix)

    # Get the highest cosine similarity for intent
    most_similar_index = cosine_similarities.argmax()
    highest_similarity = cosine_similarities[0, most_similar_index]

    best_st_response = small_talk_similarity(user_response_of_intent, highest_similarity)
    # print('I think intent is...', pred_intent, highest_similarity)
    # If small talk cosine is higher than intent cosine:
    if best_st_response != None:
        print("Neutrino:", best_st_response)
        return ['small talk']
    if pred_intent == ['exit'] and highest_similarity > 0.7: return pred_intent
    if pred_intent == ['name'] and highest_similarity > 0.6: return pred_intent
    if pred_intent == ['symptoms'] and highest_similarity > 0.4: return pred_intent
    if pred_intent == ['appointment'] and highest_similarity > 0.6: return pred_intent
    if pred_intent == ['information'] and highest_similarity > 0.5: return pred_intent
    else: return 'NOT FOUND'


def classifier_evaluation():
    """
    Evaluates the classifier using test split data, and prints the results in a confusion matrix
    """

    df = pd.read_csv('Datasets/intentmatch_dataset.csv',encoding='latin-1')

    X_train, X_test, y_train, y_test = train_test_split(df['Question'], df['Intent'], stratify = df['Intent'],test_size=0.2, random_state=42)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer(use_idf=True, sublinear_tf=True).fit(X_train_counts)
    X_train_tf = tfidf_transformer.transform(X_train_counts)




    clf = DecisionTreeClassifier(random_state=0).fit(X_train_tf, y_train)
    X_new_counts = count_vect.transform(X_test)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    print(y_test,X_test)
    y_pred = clf.predict(X_new_tfidf)

    print(multilabel_confusion_matrix(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)

    cm_disp = ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
    # Needs matplotlib installed to run
    cm_disp.plot()
    plt.show()
    cm_disp.plot()


