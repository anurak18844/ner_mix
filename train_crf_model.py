import pandas as pd
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split
import sklearn_crfsuite
from sklearn_crfsuite.utils import flatten
from sklearn import metrics as skmetrics
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../preparedata/data_train.csv")

def doc2features(doc, i):
    word = doc[i][0]
    postag = doc[i][1]
    features = {
        'word.word': word,
        'word.isspace':word.isspace(),
        'postag':postag,
        'word.isdigit()': word.isdigit()
    }
    if i > 0:
        prevword = doc[i-1][0]
        postag1 = doc[i-1][1]
        features['word.prevword'] = prevword
        features['word.previsspace'] = prevword.isspace()
        features['word.prepostag'] = postag1
        features['word.prevwordisdigit'] = prevword.isdigit()
    else:
        features['BOS'] = True
    if i < len(doc)-1:
        nextword = doc[i+1][0]
        postag1 = doc[i+1][1]
        features['word.nextword'] = nextword
        features['word.nextisspace'] = nextword.isspace()
        features['word.nextpostag'] = postag1
        features['word.nextwordisdigit'] = nextword.isdigit()
    else:
        features['EOS'] = True
    return features

def extract_features(doc):
    return [doc2features(doc, i) for i in range(len(doc))]

def df_to_nested_list(df):
    nested_list = []
    current_list = []
    
    for _, row in df.iterrows():
        word, pos, tag, boundary = row['WORD_TOKENIZE'], row['POS'], row['TAG'], row['BOUNDARY']
        current_list.append((word, pos, tag))
        
        if boundary == 'END':
            nested_list.append(current_list)
            current_list = []
    return nested_list

def get_labels(doc):
    return [tag for (token,postag,tag) in doc]
    
doc = df_to_nested_list(df)
X_data = [extract_features(d) for d in doc]
y_data = [get_labels(d) for d in doc] 

file_name = "crf_model_ner_v2"
X, X_test, y, y_test = train_test_split(X_data, y_data, test_size=0.3)
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=500,
    all_possible_transitions=True,
    model_filename=file_name
)

crf.fit(X, y)
labels = list(crf.classes_)
labels.remove('O')
y_pred = crf.predict(X_test)
e=metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels, zero_division=0)
print(e)
sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)

y_test_flat = flatten(y_test)
y_pred_flat = flatten(y_pred)
print(skmetrics.classification_report(y_test_flat, y_pred_flat, labels=sorted_labels, zero_division=0))

sorted_labels = ['B-COMMAND_1', 'I-COMMAND_1', 'B-COMMAND_2','I-COMMAND_2' 'B-FOOD', 'I-FOOD', 'B-QUESTION', 'I-QUESTION', 'B-TABLE', 'I-TABLE']
print(classification_report(y_test_flat, y_pred_flat, labels=sorted_labels, zero_division=0))

cm = confusion_matrix(y_test_flat, y_pred_flat, labels=sorted_labels)

cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

plt.figure(figsize=(10, 8))
sns.heatmap(cm_percentage, annot=True, fmt='.1f', xticklabels=sorted_labels, yticklabels=sorted_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix as Percentage (%)')
plt.savefig('confusion_matrix_crf_model.png')
plt.show()

