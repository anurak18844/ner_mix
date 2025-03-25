import speech_recognition as sr
recog = sr.Recognizer() 

qty_of_file = 286
with open("data.txt", "w", encoding="utf-8") as file:
    for i in range(qty_of_file):
        audio_path = f"speech_audio_{i+1}.wav"
        try:
            with sr.AudioFile(audio_path) as source:
                audio = recog.record(source)
            txt = recog.recognize_google(audio, language='th')
            file.write(f"{txt}\n")
            print(f"Write {txt} to {file.name}")
        except sr.UnknownValueError as e:
            print(f"error : {e}")