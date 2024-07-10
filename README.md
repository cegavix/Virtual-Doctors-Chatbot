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
Also, since I used colab for a lot of training some of the versions I used are assumed, but fingers crossed those are all correct.

## Run Instructions
 
RULE-BASED CHATBOT: Run the main.py in the relevant folder for the rule-based chatbot.
 
TRANSFER LEARNING CHATBOT: Run BERTQ_model.ipynb to create the model for use. Then, run app.py to host it in browser.
 
I had some trouble aligning the versions of  torch and transformers to make sure everything is up to date in the virtual environment, but
 
! pip install -U accelerate
! pip install -U transformers
 
Should hopefully fix that.
## Video Submission

[![youtube_video](https://img.youtube.com/vi/gMZYHU0tYn0/0.jpg)](https://www.youtube.com/watch?v=gMZYHU0tYn0)


### Demo Videos

https://github.com/cegavix/Virtual-Doctors-Chatbot/assets/98366333/ac5d795b-cb9c-471a-aba0-a4a01195d6ef


https://github.com/cegavix/Virtual-Doctors-Chatbot/assets/98366333/76301717-93b5-4081-8b5b-e5d244e9ca6e

