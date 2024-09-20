from transformers import BertModel, BertTokenizer


def get_bert(bert_path = "../pretrained_models/bert_base_chinese"):
    # Load pre-trained BERT model and tokenizer
    model = BertModel.from_pretrained(bert_path)
    tokenizer = BertTokenizer.from_pretrained(bert_path)

    return model, tokenizer

def get_sentences_embeddings(sentences):
    # Load pre-trained BERT model and tokenizer
    bert_path = "../pretrained_models/bert_base_chinese"
    model = BertModel.from_pretrained(bert_path)
    tokenizer = BertTokenizer.from_pretrained(bert_path)

    # Get token embeddings from the BERT model
    inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    sentence_embeddings = outputs.last_hidden_state.mean(dim=1)
    
    return sentence_embeddings

if __name__ == '__main__':
   text=["题目一：如图几何图形．此图由三个半圆构成，三个半圆的直径分别为直角三角形$ABC$的斜边$BC$, 直角边$AB$, $AC$.$\bigtriangleup ABC$的三边所围成的区域记为$I$,黑色部分记为$II$, 其余部分记为$III$.在整个图形中随机取一点，此点取自$I,II,III$的概率分别记为$p_1,p_2,p_3$,则$\SIFChoice$$\FigureID{1}$"]
   print(get_sentences_embeddings(text).shape)