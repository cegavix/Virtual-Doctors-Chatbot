import pickle
import random

import pandas as pd
import json

import torch
from transformers import BertForSequenceClassification, BertTokenizer

def get_prediction(text):
    # Encode the sentence with special tokens and padding
    encoded_dict = tokenizer.encode_plus(
      text,
      add_special_tokens=True,
      max_length=64,
      padding='longest',
      return_attention_mask=True,
      return_tensors="pt",
      truncation=True,
    )

    # Extract input IDs and attention mask
    input_ids = encoded_dict["input_ids"]
    attention_mask = encoded_dict["attention_mask"]

    # Pass the input through the model
    with torch.no_grad():
        outputs = model(**{"input_ids": input_ids, "attention_mask": attention_mask})

    # Get logits (prediction scores) for each class
    logits = outputs.logits.squeeze(0)  # Remove batch dimension

    # Get the predicted class label (argmax)
    predicted_class = torch.argmax(logits).item()
    pred = [predicted_class]
    prediction = label_encoder.inverse_transform(pred)[0]
    print("I think intent is:",prediction)
    # get the item index of where tag is
    index = labels.index(prediction)
    text = responses[index]

    # handle multiple items in
    if len(text) == 1:
      return text[0]
    else:
      return random.choice(text)


with open('Transfer_learning_Chatbot/BERT/label_encoder.pkl', 'rb') as file:
  try:
    # Attempt to load the label encoder
    label_encoder = pickle.load(file)
    print("Label encoder loaded successfully!")
  except EOFError:
    print("Error: Pickle file seems to be empty.")
  except pickle.UnpicklingError as e:
    print("Error loading pickle file:", e)

model = BertForSequenceClassification.from_pretrained("Transfer_learning_Chatbot/BERT/bert_model")
tokenizer = BertTokenizer.from_pretrained("Transfer_learning_Chatbot/BERT/bert_model")


with open("Transfer_learning_Chatbot/BERT/TUSHARKHETE_intents.json", "r") as read_file:
    data = (json.load(read_file))['intents']

labels = []
responses = []
for item in data:
    labels.append(item['tag'])
    responses.append(item['responses'])

# while True:
#     user_input = input("YOU:")
#     response = get_prediction(user_input)
#     print(response)

