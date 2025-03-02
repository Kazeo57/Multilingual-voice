from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai 
import speech_recognition as sr
from pydub import AudioSegment
import os
google_api_key=os.getenv('GOOGLE_API_KEY')
app = FastAPI()
origins= [
    "http://localhost:5173",
    "https://multilingualvoice.vercel.app"
]

# Ajout du middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Liste des origines autorisées
    allow_credentials=True,
    allow_methods=["*"],  # Autoriser toutes les méthodes HTTP
    allow_headers=["*"],  # Autoriser tous les headers
)


genai.configure(api_key=google_api_key)
llm=genai.GenerativeModel('gemini-1.5-flash')
# Fonction pour la transcription vocale



def convert_to_wav(audio_file_path):
    # Ouvrir le fichier audio (supposons qu'il est en MP3 par exemple)
    audio = AudioSegment.from_file(audio_file_path)
    # Sauvegarder en WAV
    wav_path = "/tmp/converted_audio.wav"
    audio.export(wav_path, format="wav")
    return wav_path


def transcribe_audio(audio_file):
    if not audio_file.lower().endswith(".wav"):
        audio_file = convert_to_wav(audio_file)
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        transcript = recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        transcript = "Sorry, I couldn't understand the audio."
    except sr.RequestError as e:
        transcript = f"Could not request results from Google Speech Recognition service; {e}"
    return transcript


# Fonction pour la traduction avec OpenAI
def translate_text(text, target_language="fr"):
    prompt = f"Translate this into {target_language}: {text}"
    
    return llm.generate_content(prompt).text

# Route API pour traiter l'audio

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    # Utilise /tmp pour stocker le fichier temporaire
    temp_file_path = "/tmp/temp.wav"
    with open(temp_file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # Utiliser le chemin temporaire pour la transcription
    transcript = transcribe_audio(temp_file_path)
    translation = translate_text(transcript)

    return {"transcription": transcript, "translation": translation}


# Lancer le serveur FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
