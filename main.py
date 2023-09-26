import base64
import datetime
import json
import sys

import requests
import speech_recognition
import speech_recognition as sr
import torch
from pydub import AudioSegment
from pydub.playback import play
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

print("Starting...")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print("Loading Large Zero-Shot Classification Model...")
context_model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
context_tokenizer = AutoTokenizer.from_pretrained(context_model_name)
context_model = AutoModelForSequenceClassification.from_pretrained(context_model_name)
def speak(message: str):
    url = "https://api.convai.com/tts/"

    payload = json.dumps({
        "transcript": message,
        "voice": "WUMale 5",
        "encoding": "mp3"
    })
    headers = {
        'CONVAI-API-KEY': 'c59f0a48ab93abab68aa38cb3f0b97f5',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    with open('audioResponse.wav', 'wb') as f:
        f.write(response.content)
    audio = AudioSegment.from_file("audioResponse.wav")
    play(audio)


def checkJarvisContext(message):
    classifier = pipeline("zero-shot-classification", model=context_model, tokenizer=context_tokenizer)
    candidate_labels = ["Talking To Jarvis", "Talking Of Jarvis", "Talking To Other"]
    result = classifier(message, candidate_labels)
    return result['labels'][0] == "Talking To Jarvis"


def checkIntent(message):
    classifier = pipeline("zero-shot-classification", model=context_model, tokenizer=context_tokenizer)
    candidate_labels = ["operation", "message"]
    result = classifier(message, candidate_labels)
    return "ACTION" if result['scores'][result['labels'].index('operation')] > result['scores'][
        result['labels'].index('message')] else "MESSAGE"

def checkRequest(message):
    classifier = pipeline("zero-shot-classification", model=context_model, tokenizer=context_tokenizer)
    candidate_labels = [
        "Set a timer",
        "Play music",
        "Turn on lights",
        "Turn off lights",
        "Check the weather",
        "Send a message",
        "Call a contact",
        "Read the news",
        "Set a reminder",
        "Find a restaurant",
        "Tell a joke",
        "Translate a phrase",
        "Search the web",
        "Open an app",
        "Answer a question",
        "Schedule an event",
        "Get directions",
        "Set an alarm",
        "Calculate a math problem",
        "Provide recommendations",
        "Tell a story",
        "Control smart home devices",
        "Create a shopping list",
        "Check your calendar",
        "Play a game",
        "Help with cooking",
        "Give sports scores",
        "Provide health information",
        "Recommend movies or TV shows",
        "Book an appointment",
        "Set a reminder",
        "Order food",
        "Check flight status",
        "Control the thermostat",
        "Provide driving directions",
        "Find nearby places",
        "Give stock market updates",
        "Manage your tasks",
        "Provide trivia",
        "Translate languages",
        "Provide historical information",
        "Perform unit conversions",
        "Manage your to-do list",
        "Answer general knowledge questions",
    ]
    result = classifier(message, candidate_labels)
    return result['labels'][0]



def makeRequest(message):
    url = "https://api.convai.com/character/getResponse"

    headers = {
        'CONVAI-API-KEY': 'c59f0a48ab93abab68aa38cb3f0b97f5'
    }
    payload = {
        'userText': message,
        'charID': 'cce5551e-59c2-11ee-9a94-42010a40000b',
        'voiceResponse': 'True'
    }
    response = requests.request("POST", url, headers=headers, data=payload)

    data = response.json()

    decode_string = base64.b64decode(data["audio"])

    with open('audioResponse.wav', 'wb') as f:
        f.write(decode_string)


def start():
    checkContext = True
    while True:
        response = takeCommand(checkContext)
        if response == "UNKNOWN_RESPONSE":
            checkContext = True
            continue
        if response == "NO_JARVIS_PREFIX":
            checkContext = True
            continue
        print("Generating response...")
        checkContext = False

        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%I:%M %p")

        intent = checkIntent(response)
        if intent == "MESSAGE":
            makeRequest(response)
            audio = AudioSegment.from_file("audioResponse.wav")
            play(audio)
        elif intent == "ACTION":
            if ("shutdown" in response):
                speak("Goodbye, Sir.")
                sys.exit()


def takeCommand(checkContext: bool):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=9)
        except speech_recognition.WaitTimeoutError:
            print("Restarting with context required...")
            return "UNKNOWN_RESPONSE"
        try:
            print("Recognizing...")
            query: str = r.recognize_google(audio, language="en")
            if checkContext:
                if not (checkJarvisContext(query)):
                    print("Condition Unmet: Query is not to J.A.R.V.I.S")
                    return "NO_JARVIS_PREFIX"
            print(f'User: {query}')
        except Exception:
            print("Unknown Response")
            return "UNKNOWN_RESPONSE"
        return query

while True:
    i = input("> ")
    print(checkRequest(i))

if __name__ == "_a_main__":
    start()
