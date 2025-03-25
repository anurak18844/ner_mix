from pythainlp.tag import pos_tag
from pythainlp.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import re
import codecs
import pandas as pd

data_not = []
def Unique(p):
    text = re.sub("\[(.*?)\]", "", p)
    text = re.sub("\[\/(.*?)\]", "", text)
    if text not in data_not:
        data_not.append(text)
        return True
    else:
        return False

def get_data(fileopen):
    with codecs.open(fileopen, 'r', encoding='utf-8-sig') as f:
        lines = f.read().splitlines()
    return [a for a in lines if Unique(a)]

def toolner_to_tag(text):
    text = text.strip()
    text = re.sub("(\[\/(.*?)\])", "\\1***", text)
    text = re.sub("(\[\w+\])", "***\\1", text)
    text2 = []
    for i in text.split('***'):
        if "[" in i:
            text2.append(i)
        else:
            text2.append("[word]"+i+"[/word]")
    text = "".join(text2)
    return text.replace("[word][/word]", "")

def postag(text):
    listtxt = [i for i in text.split('\n') if i!='']
    # print(listtxt[0])
    list_word = []
    for data in listtxt:
        list_word.append(data.split('\t')[0])
    list_word=pos_tag(list_word,engine="perceptron")
    text=""
    i=0
    for data in listtxt:
        text+=data.split('\t')[0]+'\t'+list_word[i][1]+'\t'+data.split('\t')[1]+'\t'+data.split('\t')[2]+'\n'
        i+=1
    
    return text
        
pattern = r'\[(.*?)\](.*?)\[\/(.*?)\]'
tokenizer = RegexpTokenizer(pattern)

def text2conll2002(text,pos=True):
    text = toolner_to_tag(text)
    text = text.replace("''", '"')
    text = text.replace("’", '"').replace("‘", '"')
    tag = tokenizer.tokenize(text)
    conll2002=""
    l = len(tag)
    for j, (tagopen, text, tagclose) in enumerate(tag):
        word_cut = word_tokenize(text,keep_whitespace=False)
        txt5 = ""
        for i, word in enumerate(word_cut):
            if word in ["''", '"']:
                continue
            if i == 0 and j == 0:
                if tagopen != 'word':
                    txt5 += f"{word}\tB-{tagopen}\tSTART\n"
                else:
                    txt5 += f"{word}\tO\tSTART\n"
            else:
                if tagopen != 'word':
                    if l  == j+1 and i == len(word_cut) - 1:
                        if i == 0:
                            txt5 += f"{word}\tB-{tagopen}\tEND\n"
                        else:
                            txt5 += f"{word}\tI-{tagopen}\tEND\n"
                    else:
                        if i == 0:
                            txt5 += f"{word}\tB-{tagopen}\tNone\n"
                        else:
                            txt5 += f"{word}\tI-{tagopen}\tNone\n"
                else:
                    if l  == j+1 and i == len(word_cut) - 1:
                        txt5 += f"{word}\tO\tEND\n"
                    else:
                        txt5 += f"{word}\tO\tNone\n"
        conll2002 += txt5
    if pos == False:
        return conll2002
    return postag(conll2002)
    
def alldata_list(lists):
    data_all = []
    for data in lists:
        data_num = []
        try:
            txt = text2conll2002(data,pos=True).split('\n')
            for d in txt:
                tt = d.split('\t')
                if d != "":
                    data_num.append((tt[0], tt[1], tt[2], tt[3]))
            data_all.append(data_num)
        except:
            print(data)
    return data_all

file_name = "data_tag"
data_list = get_data(file_name+".txt.")
data_list_file = alldata_list(data_list)

columns = ["WORD_TOKENIZE", "POS", "TAG", "BOUNDARY"]
df = pd.DataFrame(columns=columns)

for idx, item in enumerate(data_list_file):
    temp_df = pd.DataFrame(item, columns=columns)
    df = pd.concat([df, temp_df], ignore_index=True)

sentence_number = 1
start_idx = df[df['BOUNDARY'] == 'START'].index
end_idx = df[df['BOUNDARY'] == 'END'].index

if not start_idx.empty and not end_idx.empty:
    for start, end in zip(start_idx, end_idx):
        df.loc[start:end, 'SENTENCE'] = f"Sentence: {sentence_number}"
        sentence_number += 1

df.to_csv("data_train.csv", index=False)