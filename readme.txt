1. speech.wav -> "voice_to_text.py" -> data.txt
2. Tag for NER -> data_tag.txt
3. data_tag.txt -> "prepare_data.py" -> data_train.csv
4. data_train.csv -> "train_crf_model.py" -> model