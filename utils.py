import time

import pyautogui

from constants import intents


def search(query):
    pyautogui.hotkey("win")
    time.sleep(0.1)
    pyautogui.typewrite(query)
    time.sleep(0.1)
    pyautogui.press("enter")


# Minimum similarity threshold
min_similarity = 0.6


# Function to determine the intent based on similarity
def get_intent(nlp, statement):
    statement = nlp(statement)
    best_intent = None
    best_similarity = min_similarity  # Initialize with the minimum threshold

    for intent, data in intents.items():
        intent_descriptions = [nlp(desc) for desc in data["descriptions"]]

        # Calculate the maximum similarity between the statement and intent descriptions
        similarity = max(statement.similarity(desc) for desc in intent_descriptions)

        if similarity > best_similarity:
            best_intent = intent
            best_similarity = similarity

    if best_intent is not None:
        return best_intent, intents[best_intent]["descriptions"]
    else:
        return None, None


# Function to extract entities from the statement
def get_entities(nlp_entities, statement):
    doc_entities = nlp_entities(statement)
    entities = [(ent.text, ent.label_) for ent in doc_entities.ents]
    return entities if entities else None
