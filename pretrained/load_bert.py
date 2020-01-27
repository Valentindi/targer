import tensorflow.compat.v1 as tf
import torch
from pytorch_transformers.modeling_bert import BertConfig, BertForPreTraining, load_tf_weights_in_bert

with tf.compat.v1.Session() as sess:
    saver = tf.train.import_meta_graph('./uncased_L-12_H-768_A-12/bert_model.ckpt.meta')
    saver.restore(sess, "./uncased_L-12_H-768_A-12/bert_model.ckpt")

tf_checkpoint_path = "./uncased_L-12_H-768_A-12/bert_model.ckpt"
bert_config_file = "./uncased_L-12_H-768_A-12/bert_config.json"
pytorch_dump_path = "./uncased_L-12_H-768_A-12/pytorch_model"

config = BertConfig.from_json_file(bert_config_file)
print("Building PyTorch model from configuration: {}".format(str(config)))
model = BertForPreTraining(config)

# Load weights from tf checkpoint
load_tf_weights_in_bert(model, config, tf_checkpoint_path)

# Save pytorch-model
print("Save PyTorch model to {}".format(pytorch_dump_path))
torch.save(model.state_dict(), pytorch_dump_path)
