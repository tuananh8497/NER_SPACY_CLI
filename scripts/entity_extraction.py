import spacy
import typer
import srsly
import os
from pathlib import Path
from spacy.util import get_words_and_spaces
from collections import Counter
from prettytable import PrettyTable
import glob 
import json
import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

nlp = spacy.load("./training/model-best") 

def main(input_path: str, output_path: str, summary_path: str):
    nlp.add_pipe('sentencizer')
    all_txt = input_path + "*.txt"
    # Read all file txt in OCR and turn to lower case for normalization 
    files = glob.glob(all_txt)
    for file in files:
        file = file.replace("\\", "/")
        with open (file, "r", encoding="utf-8") as f:
            text = f.read()
            processed_text = text.lower()
            doc = nlp(processed_text)
            to_json = convert_doc(doc)
            write_data(output_name(file, output_path), to_json) # write entity extracted to json file
            write_data(output_name(file, summary_path), frequency_summarise(to_json))

def load_data(file):
    with open (file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return(data)

def write_data(file, data):
    with open (file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def output_name(file_name, path):
    file_name =  Path(file_name).stem
    path_name = path + file_name + ".json"
    return (path_name)

def convert_doc(doc):
    final = []
    summarise = []

    for sent in doc.sents:
        final_ents = []
        final_sent = sent.text

        # remove any end of line or tab in the sentence text
        final_sent = final_sent.replace('\n','') 
        final_sent = final_sent.replace('\t','')
        for ent in sent.ents:
            text, start, end, label = ent.text, ent.start_char-ent.sent.start_char, ent.end_char-ent.sent.start_char, ent.label_
            final_ents.append((text, start, end, label))
        if final_ents:
            entities = {"entities": final_ents}
            final.append([final_sent, entities])

    return (final)

def frequency_summarise(json_file):
    write_json = []
    final = []
    entity_in_word = ""

    words = [text[1]['entities'][0][0] for text in json_file]
    
    for word in words:
        entity_in_word = entity_in_word + ", " + word

    further_nlp = nlp(entity_in_word)
    test_freq = [token.text for token in further_nlp if token.is_stop != True and token.is_punct != True]
    word_freq = Counter(test_freq)
    common_words = word_freq.most_common(10)
    print(common_words)
    for word in common_words:
        write_json = {'Word': word[0],'Frequency': word[1]}
        final.append(write_json)
    return(final)
                
if __name__ == "__main__":
    typer.run(main)

