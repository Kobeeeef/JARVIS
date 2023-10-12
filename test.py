import json
import time

import g4f

import data_util

messages = data_util.load_data("messages.json")

if not messages:
    messages = []


def addSystemMessage(message):
    messages.append({"role": "system", "content": message})
    data_util.save_data("messages.json", messages)


def addAssistantMessage(message):
    messages.append({"role": "assistant", "content": message})
    data_util.save_data("messages.json", messages)


def addUserMessage(message):
    messages.append({"role": "user", "content": message})
    data_util.save_data("messages.json", messages)


# addSystemMessage(
#     "You are a intent recognition AI. You will only respond in a JSON format. with these keys: The Intent, "
#     "Need Action (boolean if external code needs to be done to DO the intent), The Entities, and the thought (What I "
#     "need to do to succeed this intent.), The message as if the intent succeed, The message as if the intent fails. "
#     "YOU MUST ONLY RESPOND IN JSON NO MATTER WHAT. The messages you respond with must always have \"sir\" as you are "
#     "J.A.R.V.I.S from Iron Man, providing intelligent assistance and responding in a manner befitting the character's "
#     "personality. act like it. Your owner is Kobe Lei. Respond with ONE JSON OBJECT.")
# addSystemMessage("You must be very helpful. and responses in messages must but lengthy.")
# addSystemMessage("ONLY RESPOND IN ONE JSON OBJECT. NO MORE!")
#
# addUserMessage("Hello")
#
# addAssistantMessage('''{
#   "Intent": "Greeting",
#   "Need Action": false,
#   "Entities": {},
#   "thought": "To respond with a friendly greeting",
#   "message": "Hello, sir!",
#   "failure_message": ""
# }''')


def getResponse(message):
    t = time.time()
    print("Requesting..")
    response = makeRequest()
    addUserMessage(message)
    addAssistantMessage(json.dumps(response, indent=4))
    print(f"Responded in: {time.time() - t} ms")
    return response


def makeRequest():
    try:
        r = g4f.ChatCompletion.create(
            model=g4f.models.gpt_35_turbo,
            messages=messages,
            provider=g4f.Provider.Aivvm
        )
        start_index = r.find('{')

        if start_index != -1:
            # Initialize counters to keep track of braces
            open_braces = 1
            close_braces = 0
            end_index = start_index + 1

            # Iterate through the string to find the matching closing curly brace
            while end_index < len(r):
                if r[end_index] == '{':
                    open_braces += 1
                elif r[end_index] == '}':
                    close_braces += 1

                # If we have found a matching closing curly brace, break the loop
                if open_braces == close_braces:
                    break

                end_index += 1

            # Extract the first JSON object
            first_json_object = r[start_index:end_index + 1]
            parsed = json.loads(first_json_object)
            if parsed is None:
                raise Exception
            if parsed == "None":
                raise Exception
            return parsed
        else:
            raise Exception
    except Exception:
        makeRequest()
