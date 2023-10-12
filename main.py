import base64
import datetime
import re
import threading

import pyttsx3
import requests
import spacy
import speech_recognition
import speech_recognition as sr
import torch
from pydub import AudioSegment
from pydub.playback import play
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

import utils
from test import getResponse

print("Starting...")
AudioSegment.converter = "ffmpeg.exe"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print("Loading Large Zero-Shot Classification Model...")
context_model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
context_tokenizer = AutoTokenizer.from_pretrained(context_model_name, use_fast=True)
context_model = AutoModelForSequenceClassification.from_pretrained(context_model_name)

print("Loading Large Natural Language Processing Model.")
nlp = spacy.load("en_core_web_lg")


def speak(message: str):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    engine.setProperty('rate', 190)
    engine.say(message)
    engine.runAndWait()


def checkJarvisContext(message):
    if not ("jarvis" in message.lower()):
        return False
    classifier = pipeline("zero-shot-classification", model=context_model, tokenizer=context_tokenizer)
    candidate_labels = ["Talking To Jarvis", "Talking About Jarvis", "Talking To Someone Else"]
    result = classifier(message, candidate_labels)

    target_label = "Talking To Jarvis"
    threshold = 0.055

    target_score = None
    result_label = None

    for label, score in zip(result['labels'], result['scores']):
        if label == target_label:
            target_score = score

    for label, score in zip(result['labels'], result['scores']):
        if label != target_label and score >= target_score + threshold:
            result_label = label
            break

    if result_label is None:
        result_label = target_label

    return result_label == "Talking To Jarvis"


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

    with open('Audios/response.wav', 'wb') as f:
        f.write(decode_string)
    return data['text']


ding_audio = AudioSegment.from_file("Audios/ding.wav")


def play_ding():
    play(ding_audio)


response_audio = AudioSegment.from_file("Audios/response.wav")


def play_response():
    play(response_audio)


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
        print(f'User: {response}')
        print("Generating response...")
        checkContext = False
        audio_thread = threading.Thread(target=play_ding)
        audio_thread.start()

        handleQuery(response)

        # utils.search(response.lower().replace("jarvis", "").replace("open", ""))


def takeCommand(checkContext: bool, timeout=0, phrase_time_limit=15):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, 0.2)
        print("Listening to Audio...")
        r.pause_threshold = 1
        try:
            audio = r.listen(source, timeout, phrase_time_limit)
        except speech_recognition.WaitTimeoutError:
            print("Restarting with context required...")
            return "UNKNOWN_RESPONSE"
        try:
            print("Transcribing audio...")
            query: str = r.recognize_google(audio, language="en")
            # query: str = r.recognize_whisper(audio, translate=True, model="tiny")
            if checkContext:
                if not (checkJarvisContext(query)):
                    print("Condition Unmet: Query is not to J.A.R.V.I.S")
                    return "NO_JARVIS_PREFIX"
        except Exception:
            print("Unknown Response")
            return "UNKNOWN_RESPONSE"
        return query


def handleQuery(query):
    intent, description = utils.get_intent(nlp, query)

    if intent is not None:
        entities = utils.get_entities(nlp, re.sub(r'jarvis', '', query, flags=re.IGNORECASE))
        entity_data = [(entity, label) for entity, label in entities] if entities else None
        if entity_data:
            entity_string = ", ".join([f"Label: {e[1]} Entity: {e[0]}" for e in entity_data])
        else:
            entity_string = None
        print(f"Intent: {intent}")
        print(f"Thought: {description[0]}")
        print(f"Entities: {entity_string}")
        if intent == "time":
            first_gpe = next((entity for entity, label in entity_data if label == "GPE"), None)
            if first_gpe:
                current_time = datetime.datetime.now()
                formatted_time = current_time.strftime("%I:%M %p")
                speak(f"Sir, the time in {first_gpe} is {formatted_time}")
            else:
                current_time = datetime.datetime.now()
                formatted_time = current_time.strftime("%I:%M %p")
                speak(f"Sir, the time is {formatted_time}")
    else:
        text = makeRequest(query)
        audio = AudioSegment.from_file("Audios/response.wav")
        print(f"Response: {text}")
        play(audio)


if __name__ == "__main__":
    start()
