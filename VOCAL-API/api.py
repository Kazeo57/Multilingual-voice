from fastapi import FastAPI, UploadFile, File
import google.generativeai as genai 
import speech_recognition as sr
import os
google_api_key=os.getenv('GOOGLE_API_KEY')
app = FastAPI()

genai.configure(api_key=google_api_key)
llm=genai.GenerativeModel('gemini-1.5-flash')
# Fonction pour la transcription vocale
def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    return recognizer.recognize_google(audio)

# Fonction pour la traduction avec OpenAI
def translate_text(text, target_language="fr"):
    prompt = f"Translate this into {target_language}: {text}"
    
    return llm.generate_content(prompt).text

# Route API pour traiter l'audio
@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    with open("temp.wav", "wb") as buffer:
        buffer.write(await file.read())
    
    transcript = transcribe_audio("temp.wav")
    translation = translate_text(transcript)

    return {"transcription": transcript, "translation": translation}

# Lancer le serveur FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
