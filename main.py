import pandas as pd
from joblib import load
import nltk
from util import set_name


def predict_with_confidence(symptoms):
    # Use the model to predict the illness with a measure of confidence
    classifier = load('symptom_classifier.joblib')
    prediction = classifier.predict(symptoms)
    fitted_svm_model = classifier.estimators_[1]

    # TODO: Would also be good to check they are in the same COLUMN name (e.g 23 for disease 23) and the prediction (23) are the same
    svm_proba_val = fitted_svm_model.predict_proba(symptoms)
    max_proba = svm_proba_val.max(axis=1)

    if max_proba > 0.6:
        return prediction
    elif max_proba > 0.4:
        return prediction, 'NOT CONFIDENT'
    else:
        return 'DONT KNOW'


if __name__ == "__main__":
    user_name = 'User'
    flag = True
    print(
        "Disclaimer: Please keep in mind that this is a form of personalised suggestion healthcare. Any diagnoses or advice are only suggestions and should NOT be taken as fact until verified by  an actual GP. If you are experiencing a medical emergency, please call 999 immediately.")

    user_input = input(
        "I am Neutrino, your ChatBot GP. In this virtual doctor's appointment, I will listen to your symptoms, potentially form a diagnosis and suggest treatments if possible. What is your name?")
    user_name = set_name(user_input)
    print("Neutrino: Hello %s, it is nice to meet you! How can I help you today?" % user_name)

    while flag:
        user_input = input('%s: ' % user_name)

        intent = classify_intent_similarity(user_input, user_name)

        if intent == ['small talk']:
            pass
        elif intent == ['exit']:
            print("Have a nice day. Goodbye!")
            break
        elif intent == ['name']:
            change_name = input(
                'Neutrino: Your name is %s. Would you like to change it? Please say Yes or No:' % user_name).lower()
            if change_name == 'yes':
                user_name = set_name(input('What is your name?'))
            print('Neutrino: Okay, %s. What do you want to talk about now?' % user_name)
        elif intent == ['symptoms']:
            print("Neutrino: In order to figure out what might be wrong, I need to know your symptoms.")
            ask_symptoms(user_input, user_name)
        elif intent == ['appointment']:
            print('Neutrino: Checking for appointments...')
            check_appointment(user_input, user_name)
        elif intent == ['question']:
            pass
        else:
            # TODO: make an array of generic acknowledgmenets (CUI)
            print("Papa: Thanks for sharing. What do you want to talk about now?")


