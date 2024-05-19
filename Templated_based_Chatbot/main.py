import random
from simple_colors import *

import pandas as pd
from joblib import load
import nltk

from make_appointment import make_appointment
from information import display_search
from intent_matching import classify_intent_similarity
from util import set_name, make_arrays_from_csv
from get_symptoms import ask_symptoms, predict_with_confidence

if __name__ == "__main__":
    user_name = 'User'
    flag = True
    print('\n \n \n')
    print(blue(
        "Disclaimer: Please keep in mind that this is personalised suggestion healthcare. Any diagnoses or "
        "advice are only suggestions and should NOT be taken as fact until verified by an actual GP. If you are "
        "experiencing a medical emergency, please call 999 immediately."))

    print('\n \n \n')
    print("--------------- W E L C O M E - T O - N E U T R I N O! --------------------------")

    user_input = input(
        "I am Neutrino, your ChatBot GP for Wisteria Manor Hospital. In this virtual doctor's appointment, I will listen to your symptoms, "
        "potentially form a diagnosis and suggest treatments I can also make you appointments. What is your name?")
    user_name = set_name(user_input)
    print(blue("Neutrino: Hello %s, it is nice to meet you! How can I help you today? If your stuck, I recommend 1) "
          "asking for a diagnosis. Then, with the information I give you, 2) Ask about treatment options and then 3) "
          "Make an appointment if necessary." % user_name))

    acknowledgements = ['Thanks for sharing. What can I do for you?', 'I see. What would you like to do now?',
                        'I understand. What would you like to do next? Symptoms, treatments or appointments.']
    prediction = ''
    print('\n \n \n')

    while flag:
        user_input = input('%s: ' % user_name)

        intent = classify_intent_similarity(user_input, user_name)
        if intent == ['small talk']:
            pass
        elif intent == ['exit']:
            print(blue("Neutrino: Have a nice day. Goodbye!"))
            break
        elif intent == ['name']:
            change_name = input(
                'Neutrino: Your name is %s. Would you like to change it? Please say Yes or No:' % user_name).lower()
            if change_name == 'yes':
                user_name = set_name(input('What is your name?'))
            print(blue('Neutrino: Okay, %s. What do you want to talk about now?' % user_name))
        elif intent == ['symptoms']:
            print('\n')
            print(blue("Neutrino: Let's figure out what the issue might be."))
            found_symptoms = ask_symptoms(user_input, [])
            prediction, confidence = predict_with_confidence(found_symptoms, False)
        elif intent == ['appointment']:
            print('\n \n \n')
            print(blue('Neutrino: Checking for appointments...'))
            notes = 'Summary of Consultation with ChatBot: Diagnosis: ' + prediction + ' Symptoms: ' + str(found_symptoms) if prediction != '' else ''
            make_appointment(user_input, user_name, notes)
            #  check_appointment(user_input, user_name, prediction, found_symptoms)
        elif intent == ['information']:
            if display_search(user_input) is None:
                if prediction != '':
                    display_search(prediction)
                    print(blue("Neutrino: I hope that information was helpful. What else would you like to know?"))
                else:
                    print(blue("Neutrino: What illness would you like more information on?"))
        else:
            # TODO: make an array of generic acknowledgmenets (CUI)
            random_acknowledgement = random.choice(acknowledgements)
            print('Neutrino:', random_acknowledgement)

