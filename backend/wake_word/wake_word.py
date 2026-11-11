import asyncio
import time
import webrtcvad
import pyaudio
import pyautogui
import pyperclip
from pathlib import Path
from livekit.wakeword import WakeWordModel, WakeWordListener

"""
        Handy is an external desktop GUI that automatically transcribes audio using your
        local model (Canary/Whisper) and places the text into your clipboard/active input field.
        Because Handy already includes a built-in Silero VAD (Voice Activity Detector),
        you don't have to manually count silence chunks. When you use Push-to-Talk, holding the hotkey keeps recording,
        and releasing it immediately finishes transcription and pastes the text.
        We can use webrtcvad in Python to monitor voice activity: once you finish speaking (1 second of silence),
        the script releases Handy's key, waits for Handy to paste/copy, and reads the text from the clipboard.

"""

MODEL_PATH = Path(r"backend\wake_word\hey_jarvis.onnx")

HANDY_KEYS = ["ctrl", "space"]
SILENCE_DURATION = 1.0  # Stop after 1s of silence


def hold_keys(keys):
    """Press down all keys in combination."""
    for key in keys:
        pyautogui.keyDown(key)


def release_keys(keys):
    """Release all keys in reverse order."""
    for key in reversed(keys):
        pyautogui.keyUp(key)


def record_until_silence(silence_limit=1.0, max_wait_for_speech=5.0):
    vad = webrtcvad.Vad(2)
    rate = 16000
    chunk = rate * 30 // 1000

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=rate,
        input=True,
        frames_per_buffer=chunk,
    )

    started = False
    silence_start = None
    start_time = time.time()

    try:
        while True:
            data = stream.read(chunk, exception_on_overflow=False)
            speech = vad.is_speech(data, rate)

            if speech:
                started = True
                silence_start = None
            elif started:
                silence_start = silence_start or time.time()

                if time.time() - silence_start >= silence_limit:
                    break
            elif time.time() - start_time >= max_wait_for_speech:
                print("No speech detected after wake word.")
                break
            time.sleep(0.01)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


async def main():
    model = WakeWordModel(models=[str(MODEL_PATH)])

    try:
        async with WakeWordListener(model, threshold=0.6, debounce=2.0) as listener:
            print("Ready! Say 'Jarvis' to wake up Handy...")

            while True:
                detection = await listener.wait_for_detection()
                print(
                    f"Detected: {detection.name} (confidence: {detection.confidence:.2f})"
                )
                print("\n[Jarvis detected] Listening to your speech...")

                # 1. Clear clipboard to detect new transcription
                pyperclip.copy("")
                # 2. Press and hold Handy recording keys
                hold_keys(HANDY_KEYS)

                try:
                    # 3. Detect when conversation cuts off (1s silence)
                    record_until_silence(silence_limit=SILENCE_DURATION)
                finally:
                    # 4. Release keys guaranteed -> Handy stops & transcribes
                    release_keys(HANDY_KEYS)
                    print("Conversation paused. Processing transcription...")

                transcription = ""
                for _ in range(50):  # poll up to 5 seconds
                    text = pyperclip.paste().strip()
                    if text:
                        transcription = text
                        break
                    time.sleep(0.1)
                if transcription:
                    print(f"Transcription: {transcription}")
                else:
                    print("No text received from Handy.")
    finally:
        print("Model Turned Off")


if __name__ == "__main__":
    asyncio.run(main())
