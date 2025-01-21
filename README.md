## A Chatbot to simulate patient-physician general practitioner (GP) appointments
Here u will find a sample transcript, demo videos, my code for generating all 4 intelligent agents (+ the decision tree) and their relevant datasets.

### Setup

1. Clone the repository
2. Install the required packages on a virtual environment
- Example bash script, for MacOS/Linux:


```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The transformers module can be really finnicky with the accelerate version, if this happens update all their dependencies too.

## Run Instructions

RULE-BASED CHATBOT: Run the main.py in the relevant folder for the rule-based chatbot.
 
TRANSFER LEARNING CHATBOT: Run BERTQ_model.ipynb to create the model for use. Then, run app.py to host it in browser.
 
ALIGN the versions of  torch and transformers to make sure everything is up to date in the virtual environment:
 
! pip install -U accelerate
! pip install -U transformers

![Screenshot 2025-01-21 at 11 59 21](https://github.com/user-attachments/assets/cbddff9e-4b22-4cae-8b43-db39cb1df8a0)

## Concepts and Theory
 ![Screenshot 2025-01-21 at 13 01 33](https://github.com/user-attachments/assets/5d0d3f46-6c1f-4c15-af09-8a1b70ebbd23)

## Use Case Diagrams of the 2 Chatbots
NLP-Based Chatbot

![Screenshot 2025-01-17 at 03 21 48](https://github.com/user-attachments/assets/1f15f55b-04a8-41d0-82d1-977d2e3a5b8c)

Transformer-Based Chatbot

[Screenshot 2025-01-17 at 03 21 23](https://github.com/user-attachments/assets/461084b9-31b1-4968-94f0-ff659c541b4b)


### Demo Videos!


https://github.com/cegavix/Virtual-Doctors-Chatbot/assets/98366333/ac5d795b-cb9c-471a-aba0-a4a01195d6ef


https://github.com/cegavix/Virtual-Doctors-Chatbot/assets/98366333/76301717-93b5-4081-8b5b-e5d244e9ca6e

## Video Submission

[![youtube_video](https://img.youtube.com/vi/gMZYHU0tYn0/0.jpg)](https://www.youtube.com/watch?v=gMZYHU0tYn0)

