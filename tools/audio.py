import logging
from langchain_core.tools import tool
import speech_recognition as sr
from pydub import AudioSegment
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@tool
def transcribe_audio(audio_path: str) -> dict:
    """Transcribe audio file to text using speech recognition.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        dict with keys: text, success
    """
    logger.info("=" * 80)
    logger.info("🎤 TRANSCRIBING AUDIO")
    logger.info("=" * 80)
    logger.info(f"Audio path: {audio_path}")
    
    try:
        if not audio_path.startswith("LLMFiles"):
            full_path = os.path.join("LLMFiles", file_path)
        else:
            full_path = audio_path

        if not os.path.exists(full_path):
            return f"Error: File not found at {full_path}"

        # 2. Convert to WAV (SpeechRecognition needs WAV)
        # We create a temporary wav filename
        wav_path = full_path + ".wav"
        
        # Load the audio file (pydub handles mp3, opus, etc. via ffmpeg)
        audio = AudioSegment.from_file(full_path)
        audio.export(wav_path, format="wav")

        # 3. Transcribe
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)

        # 4. Cleanup temporary wav file
        if os.path.exists(wav_path):
            os.remove(wav_path)

        return text
    
    except sr.UnknownValueError:
        logger.warning("⚠️ Speech recognition could not understand audio")
        return {
            "text": "",
            "success": False,
            "error": "Could not understand audio"
        }
    
    except sr.RequestError as e:
        logger.error(f"❌ Speech recognition service error: {str(e)}")
        return {
            "text": "",
            "success": False,
            "error": f"Service error: {str(e)}"
        }
    
    except Exception as e:
        logger.error(f"❌ Error transcribing audio: {str(e)}")
        logger.exception(e)
        return {
            "text": "",
            "success": False,
            "error": str(e)
        }