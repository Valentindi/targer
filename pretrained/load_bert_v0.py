#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from IPython import get_ipython

get_ipython().system(" wget 'https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip' - O 'uncased_L-12_H-768_A-12.zip'")


# In[ ]:


get_ipython().system(" unzip 'uncased_L-12_H-768_A-12.zip'")


# In[18]:


import tokenizer_custom_bert

text = "After stealing embeddings from the bank's accounts, the bank robber was seen driving on the Mississippi river bank in mini-van."
text = '[CLS]' + text + '[SEP]'
tokenizer = tokenizer_custom_bert.BertTokenizer.from_pretrained("https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt")
tokenized_text = tokenizer.tokenize(text)
print (tokenized_text)


# In[3]:


import tensorflow as tf
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./uncased_L-12_H-768_A-12/bert_model.ckpt.meta')
    saver.restore(sess, "./uncased_L-12_H-768_A-12/bert_model.ckpt")


# In[5]:


import torch

from pytorch_transformers.modeling_bert import BertConfig, BertForPreTraining, load_tf_weights_in_bert


tf_checkpoint_path="./uncased_L-12_H-768_A-12/bert_model.ckpt"
bert_config_file = "./uncased_L-12_H-768_A-12/bert_config.json"
pytorch_dump_path="./uncased_L-12_H-768_A-12/pytorch_model"

config = BertConfig.from_json_file(bert_config_file)
print("Building PyTorch model from configuration: {}".format(str(config)))
model = BertForPreTraining(config)

# Load weights from tf checkpoint
load_tf_weights_in_bert(model, config, tf_checkpoint_path)

# Save pytorch-model
print("Save PyTorch model to {}".format(pytorch_dump_path))
torch.save(model.state_dict(), pytorch_dump_path)


# In[9]:


import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

path_to_pretrained = "./uncased_L-12_H-768_A-12"
bert_model = BertModel.from_pretrained(path_to_pretrained)


# In[16]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import tokenizer_custom_bert

text = "After stealing embeddings from the bank's accounts, the bank robber was seen driving on the Mississippi river bank in mini-van."
text = '[CLS]' + text + '[SEP]'
tokenizer = tokenizer_custom_bert.BertTokenizer.from_pretrained("https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt")
tokenized_text = tokenizer.tokenize(text)
print (tokenized_text)


# In[ ]:




