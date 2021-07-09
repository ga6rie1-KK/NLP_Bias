import torch
from transformers import BertJapaneseTokenizer, BertForMaskedLM

# Load pre-trained tokenizer
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

# Tokenize input
text = 'テレビでサッカーの試合を見る。'
tokenized_text = tokenizer.tokenize(text)
# ['テレビ', 'で', 'サッカー', 'の', '試合', 'を', '見る', '。']

# Mask a token that we will try to predict back with `BertForMaskedLM`
# masked_indexs = [2,4]
masked_indexs = range(4,len(tokenized_text))
for i in masked_indexs:
    tokenized_text[i] = '[MASK]'
# ['テレビ', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]', '[MASK]']

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])

# Load pre-trained model
model = BertForMaskedLM.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
model.eval()

# Predict
with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = [outputs[0][0, i].topk(5) for i in masked_indexs] # 予測結果の上位5件を抽出

# Show results
for j in range(len(masked_indexs)):
    for i, index_t in enumerate(predictions[j].indices):
        index = index_t.item()
        token = tokenizer.convert_ids_to_tokens([index])[0]
        print(i, token)
    print('\n')
