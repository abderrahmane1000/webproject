import random
import json
import torch
import os
import requests

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Dr. Sam"
excluded_tags = ["greeting", "thanks", "goodbye"]

def count_matching_patterns(user_input, patterns):
    return sum(1 for pattern in patterns if pattern.lower() in user_input.lower())

def use_llama_model(prompt):
    print("Calling meta-llama mental health model...")

    API_URL = "https://api.groq.com/openai/v1/chat/completions"
    HEADERS = {
        "Authorization": "Bearer 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",
        "Content-Type": "application/json"
    }
    data = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [
            {"role": "system", "content": "You are a compassionate mental health professional named Dr. Sam."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(API_URL, headers=HEADERS, json=data)
        response.raise_for_status()
        reply = response.json()["choices"][0]["message"]["content"]
        return reply.strip()
    except Exception as e:
        print("Error using LLaMA API:", e)
        return "I'm here to listen. Please tell me more about how you're feeling."

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                if tag in excluded_tags:
                    reply = random.choice(intent['responses'])
                else:
                    match_count = count_matching_patterns(msg, intent["patterns"])
                    if match_count >= 5:
                        reply = random.choice(intent['responses'])
                    else:
                        reply = "Can you tell me more about what's troubling you?"
                break
    else:
        reply = use_llama_model(msg)

    return reply

if __name__ == "__main__":
    print("Start talking to Dr. Sam. Type 'quit' to end.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        reply = get_response(user_input)
        print(f"{bot_name}: {reply}")
