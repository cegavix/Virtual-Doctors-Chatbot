import pandas as pd
from joblib import load
import nltk
from util import set_name, make_arrays_from_csv
from get_symptoms import ask_symptoms, predict_with_confidence

if __name__ == "__main__":
    user_name = 'User'
    flag = True
    print(
        "Disclaimer: Please keep in mind that this is a form of personalised suggestion healthcare. Any diagnoses or "
        "advice are only suggestions and should NOT be taken as fact until verified by  an actual GP. If you are "
        "experiencing a medical emergency, please call 999 immediately.")

    user_input = input(
        "I am Neutrino, your ChatBot GP. In this virtual doctor's appointment, I will listen to your symptoms, "
        "potentially form a diagnosis and suggest treatments if possible. What is your name?")
    user_name = set_name(user_input)
    print("Neutrino: Hello %s, it is nice to meet you! How can I help you today? If your stuck, I reccommend 1) "
          "asking for a diagnosis. Then, with the information I give you, 2) ask about treatment options and then 3) "
          "make an appointment if necessary." % user_name)

    while flag:
        user_input = input('%s: ' % user_name)

        # intent = classify_intent_similarity(user_input, user_name)
        intent = ['symptoms']

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
            print("Neutrino: Sure thing! Let's figure out what the issue might be.")
            found_symptoms = ask_symptoms(user_input, [])
            prediction, confidence = predict_with_confidence(found_symptoms)
        elif intent == ['appointment']:
            print('Neutrino: Checking for appointments...')
            # TODO: make database with time, date, doctor, and patient name and automatically generated notes (where
            #  they can put the diagnosis and found symptoms)
            #  check_appointment(user_input, user_name, prediction, found_symptoms)
        elif intent == ['question']:
            pass
        else:
            # TODO: make an array of generic acknowledgmenets (CUI)
            print("Papa: Thanks for sharing. What do you want to talk about now?")


