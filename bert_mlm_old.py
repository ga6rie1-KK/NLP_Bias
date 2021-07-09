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

male_occupation = []
female_occupation = []

def bert(word):
    # preprocess
    sentence = "[MASK] is a {}.".format(word)
    model_input = bert_tokenizer.encode(sentence, return_tensors='pt')
    masked_token_index = torch.where(model_input == bert_tokenizer.mask_token_id)[1]

    # predict
    token_logits = bert_model(model_input)[0]
    masked_token_logits = token_logits[0, masked_token_index, :]
    top_100_tokens = torch.topk(masked_token_logits, 100, dim=1).indices[0].tolist()
    itr = 0
    for token in top_100_tokens:
        itr += 1
        if bert_tokenizer.decode([token]) == "he":
            prediction = "男性"
            break
        elif bert_tokenizer.decode([token]) == "she":
            prediction = "女性"
            break
    return prediction, itr

def textInput_callback(attr, old, new):
    prediction, itr = bert(new)
    if prediction == "男性":
        male_occupation.append(new)
    elif prediction == "女性":
        female_occupation.append(new)
    if len(curdoc().roots) > 1:
        for model in curdoc().roots[1:]:
            curdoc().remove_root(model)
    output_prediction = TextInput(value=prediction, title="BERTの予想")
    curdoc().add_root(output_prediction)

# make text input
txtInput = TextInput(value="", title="単語を入力")
txtInput.on_change('value', textInput_callback)
curdoc().add_root(txtInput)

# 使い方：bokeh serve --show bert_mlm.py

# sentence = "The {} tried to sleep but [MASK] couldn't sleep.".format(word)
# sentence = "The {} went to a bed and [MASK] fell asleep.".format(word)
# sentence = "As the {} was hungry, [MASK] ate lunch.".format(word)
# sentence = "[MASK] is a {}.".format(word)
# sentence = "The {} was asked a question but [MASK] didn't answered.".format(word)
