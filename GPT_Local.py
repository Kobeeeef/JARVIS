from gpt4all import GPT4All

from SpeechSynthesis import speak_pyttsx3
from Utils.Utils import get_current_time

orca = "orca-mini-3b.ggmlv3.q4_0.bin"
llama_7b = "llama-2-7b-chat.ggmlv3.q4_0.bin"
model = GPT4All(llama_7b, device="cpu")
system_template = ("<<SYS>>Your name is JARVIS, a highly advanced Virtual Voice AI assistant. You "
                   "are designed to assist with any task when allowed. Be very direct, serious, and most importantly "
                   "SHORT RESPONSES. Maintain a polite and professional tone in your responses."
                   " Prioritize contextual awareness and unwavering loyalty."
                   " You are known for having most common sense. Respond the shortest you possibility can. "
                   "No action tags.<</SYS>>")
with model.chat_session(system_template):
    while True:
        print()
        i = input("> ")
        user = "Kobe"
        time = get_current_time()
        prompt = f"Time: {time}\n{user}: {i}"
        response = model.generate(prompt=prompt, temp=0, max_tokens=500, streaming=True)
        speak_pyttsx3(response)
        for i in response:
            print(i, flush=True, end="")
