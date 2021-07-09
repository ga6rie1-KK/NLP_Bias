from transformers import AutoModelWithLMHead, AutoTokenizer
import torch
from bokeh import palettes
from bokeh.plotting import figure, output_file
from bokeh.io import output_notebook, show, curdoc
from bokeh.transform import transform
from bokeh.models import *
from bokeh.layouts import *

# define model & tokenizer
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_model = AutoModelWithLMHead.from_pretrained('bert-base-uncased')

male_occupation = set()
female_occupation = set()

sentence_for_prediction = "[MASK] is a {}."

def bert(word, input_sentence):
    # preprocess
    sentence = input_sentence.format(word)
    model_input = bert_tokenizer.encode(sentence, return_tensors='pt')
    masked_token_index = torch.where(model_input == bert_tokenizer.mask_token_id)[1]

    # predict
    token_logits = bert_model(model_input)[0]
    masked_token_logits = token_logits[0, masked_token_index, :]
    masked_token_proba = torch.nn.functional.softmax(masked_token_logits[0], dim=0)
    top_5_tokens = torch.topk(masked_token_logits, 5, dim=1).indices[0].tolist()
    top_5_tokens_proba = [masked_token_proba[i].item() for i in top_5_tokens]
    prediction = []
    proba = []
    flag = True
    for i, token in enumerate(top_5_tokens):
        if flag:
            if bert_tokenizer.decode([token]) == "he":
                male_occupation.add(word)
                flag = False
            elif bert_tokenizer.decode([token]) == "she":
                female_occupation.add(word)
                flag = False
        prediction.append(sentence.replace("[MASK]", bert_tokenizer.decode([token])))
        proba.append(top_5_tokens_proba[i])
    return prediction, proba

def textInput_callback(attr, old, new):
    prediction, proba = bert(new, sentence_for_prediction)
    out_text = []
    for i in range(len(prediction)):
        out_text.append(prediction[i] + "(" + str(round(proba[i], 4)) + ")")
    if len(curdoc().roots) > 2:
        for model in curdoc().roots[2:]:
            curdoc().remove_root(model)
    output_prediction = TextAreaInput(value="\n".join(out_text), rows=5, title="BERTの予想top5", sizing_mode='stretch_width')
    curdoc().add_root(output_prediction)

def sentenceInput_callback(attr, old, new):
    global sentence_for_prediction
    sentence_for_prediction = new.replace("[OCC]", "{}")

# make text input
sentence_input = TextInput(value=sentence_for_prediction.format("[OCC]"), title="文を入力", sizing_mode='stretch_width')
sentence_input.on_change('value', sentenceInput_callback)
txtInput = TextInput(value="", title="職業を入力", sizing_mode='stretch_width')
txtInput.on_change('value', textInput_callback)
curdoc().add_root(sentence_input)
curdoc().add_root(txtInput)

# 使い方：bokeh serve --show bert_mlm.py

# sentence = "The {} tried to sleep but [MASK] couldn't sleep.".format(word)
# sentence = "The {} went to a bed and [MASK] fell asleep.".format(word)
# sentence = "As the {} was hungry, [MASK] ate lunch.".format(word)
# sentence = "[MASK] is a {}.".format(word)
# sentence = "The {} was asked a question but [MASK] didn't answered.".format(word)
