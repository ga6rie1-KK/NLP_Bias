from transformers import AutoModelWithLMHead, AutoTokenizer
import torch
from bokeh import palettes
from bokeh.plotting import figure
from bokeh.io import curdoc
from bokeh.transform import transform
from bokeh.models import *
from bokeh.layouts import *

# define model & tokenizer
gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2')
gpt2_model = AutoModelWithLMHead.from_pretrained('gpt2')

def get_top2sentence(sentence):
    """
    input:複数の文
    output:最初の二つの文
    """
    word_list = sentence.split()
    num_period = 0
    for i, word in enumerate(word_list):
        if "\"\"" in word:
            idx = list(word).index("\"")
            first = word[:idx+1]
            second = word[idx+1:]
            word_list = word_list[:i] + [first] + [second] + word_list[i+1:]
    for i, word in enumerate(word_list):
        if "." in word:
            if  not "\"" in word:
                num_period += 1
            elif word[-1] == "\"":
                next_head = word_list[i+1][0]
                if next_head == "\"" or next_head.isupper():
                    num_period += 1
        if num_period == 2:
            last_id = i
            break
    top2sentence = word_list[0:i+1]
    return " ".join(top2sentence)

def gpt2(sentence):
    """
    input:~~~. Heの形式の続きを予測したい文(str)
    output:~~~. He ~~~.の形式の予測された文(str), 各層のattention typeごとのattentionの値
    """
    # tokenize sentence
    model_input = gpt2_tokenizer(sentence, return_tensors='pt')

    # predict sentence
    output = gpt2_model.generate(model_input['input_ids'], max_length=50)
    predicted_sentence = gpt2_tokenizer.decode(output[0])

    # get attentions from he,she to input words
    # 各層のattention typeごとのattentionの値[layer, attention type, attention]
    attentions = gpt2_model(**model_input, output_attentions=True)[2] # gpt2_model.generateと二度手間のような気がするからいつか統合したい(generate methodについて調べる)
    attention_all = torch.Tensor([[[i for i in attention[-1]] for attention in layer[0]] for layer in attentions])
    
    return predicted_sentence, attention_all

def textInput_callback(attr, old, new):
    male_prediction, male_attention = gpt2(new + " He")
    female_prediction, female_attention = gpt2(new + " She")
    if len(curdoc().roots) > 1:
        for model in curdoc().roots[1:]:
            curdoc().remove_root(model)
    txtOutput_male = TextInput(value=get_top2sentence(male_prediction), title="入力 + Heの続き", sizing_mode='stretch_width')
    curdoc().add_root(txtOutput_male)
    txtOutput_female = TextInput(value=get_top2sentence(female_prediction), title="入力 + Sheの続き", sizing_mode='stretch_width')
    curdoc().add_root(txtOutput_female)
    plot_attention(new + " He", new + " She", male_attention, female_attention)

def plot_attention(male_sentence, female_sentence, male_data, female_data):
    def plot(sentence, data):
        # preprocess
        num_layer = data.shape[0]
        num_attention = data.shape[1]
        sentence = sentence.split()
        sentence = sentence[:-2] + [sentence[-2][:-1]] + ["."] + [sentence[-1]]
        att_type = [str(num_attention-i) for i in range(num_attention)]
        x = []
        y = []
        for i in range(num_attention):
            for j in range(len(sentence)):
                x.append(sentence[j])
                y.append(str(i+1))
        df = {'x':x,
              'y':y
             }
        for k in range(num_layer):
            value = []
            for i in range(num_attention):
                for j in range(len(sentence)):
                    value.append(data[k][i][j].item())
                    df['layer{}'.format(k+1)] = value
        # source = {'x':sentence, 'y':att_type, 'layer1':attention of layer1, 'layer2':attention of layer2, ...}
        source = ColumnDataSource(df)
        colors = palettes.gray(256)
        mapper = LinearColorMapper(palette=colors, low=0, high=1)

        # make selector
        options = ["layer{}".format(i+1) for i in range(num_layer)]
        select = Select(title="Layer", value="layer1", options=options)
        def select_callback(attr, old, new):
            p.tools = []
            TOOLTIPS = [("word", "@x"),
                        ("attention type", "@y"),
                        ("attention value", "@{}".format(select.value))
                       ]
            p.add_tools(HoverTool(tooltips=TOOLTIPS))
            p.square(x='x', y='y', source=source,
                     size=40,
                     line_color=None,
                     fill_color=transform('{}'.format(select.value), mapper)
                    )

        select.on_change("value", select_callback)

        # preprocess
        TOOLTIPS = [("word", "@x"),
                    ("attention type", "@y"),
                    ("attention value", "@{}".format(select.value))
                   ]

        # plot
        p = figure(width=len(sentence)*40, height=num_attention*40,
                   x_range=sentence, y_range=att_type,
                   x_axis_location="above",
                   tools="save", tooltips=TOOLTIPS
                  )
        p.square(x='x', y='y', source=source,
                 size=40,
                 line_color=None,
                 fill_color=transform('{}'.format(select.value), mapper)
                )
        p.axis.axis_line_color = None
        p.axis.major_tick_line_color = None
        p.yaxis.axis_label = "attention type"
        layout = Column(select, p)
        return layout

    male_layout = plot(male_sentence, male_data)
    female_layout = plot(female_sentence, female_data)
    layout = Row(male_layout, female_layout)
    curdoc().add_root(layout)
    

# make text input
txtInput = TextInput(value="", title="テキストを入力")
txtInput.on_change('value', textInput_callback)
curdoc().add_root(txtInput)

# 使い方：bokeh serve --show (--port 5008) gpt2_txtGene.py