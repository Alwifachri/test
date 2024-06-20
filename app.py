from flask import Flask, request, jsonify
from transformers import BertForSequenceClassification, BertTokenizerFast, pipeline
from flask_cors import CORS
import pandas as pd
import random
import json

# Load intents dataset
def load_json_file(filename):
    with open(filename) as f:
        return json.load(f)

filename = 'dataset_chatbot.json'
intents = load_json_file(filename)

# Create DataFrame from intents
def create_df(intents):
    patterns = []
    tags = []
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            patterns.append(pattern)
            tags.append(intent['tag'])

    df = pd.DataFrame({
        'Pattern': patterns,
        'Tag': tags
    })
    return df

df = create_df(intents)

labels = df['Tag'].unique().tolist()
labels = [s.strip() for s in labels]

num_labels = len(labels)
id2label = {id: label for id, label in enumerate(labels)}
label2id = {label: id for id, label in enumerate(labels)}

model_path = "model/content/chatbot"
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
tokenizer = BertTokenizerFast.from_pretrained(model_path)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Setting up the API
app = Flask(__name__)
CORS(app)

@app.route("/chat", methods=["POST"])
def chat():
    input_text = request.json["input_text"]
    # Use the classifier pipeline to get the text classification result
    result = classifier(input_text)
    predicted_label = result[0]["label"]

    # Directly use the label if the classifier returns the correct intent
    if predicted_label in label2id:
        intent_tag = predicted_label
    else:
        # Handle potential mismatches or unexpected label formats
        intent_tag = id2label[int(predicted_label.split('_')[-1])] if predicted_label.split('_')[-1].isdigit() else 'unknown'

    # Get a random response from the corresponding intent
    response = "maaf aku tidak paham."
    for intent in intents['intents']:
        if intent['tag'] == intent_tag:
            response = random.choice(intent['responses'])
            break

    return jsonify({"response_text": response})

if __name__ == "__main__":
    app.run()
