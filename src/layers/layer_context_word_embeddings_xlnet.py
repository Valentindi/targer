"""class implements context word embeddings, like Elmo, Bert, XlNet"""
"""The meaning of the equal word can change in different context, in different batch"""

from src.layers.layer_base import LayerBase
import torch
import torch.nn.functional as F

import logging

class LayerContextWordEmbeddingsXlNet(LayerBase):
    """LayerWordEmbeddings implements word embeddings."""
    def __init__(self, word_seq_indexer, gpu, freeze_word_embeddings=False, tpnm = "XlNet", pad_idx=0, embedding_dim=768):
        super(LayerContextWordEmbeddingsXlNet, self).__init__(gpu)
        print ("LayerContextWordEmbeddings dert init")
        self.embeddings = word_seq_indexer.emb
        self.embeddings.padding_idx = pad_idx
        self.word_seq_indexer = word_seq_indexer
        self.embeddings_dim = embedding_dim
        self.output_dim = self.embeddings_dim
        self.gpu = gpu
        self.tpnm = tpnm

    def is_cuda(self):
        return self.embeddings.weight.is_cuda
    
    def to_gpu(self, tensor):
        if self.gpu > -1:
            return tensor.cuda(device=self.gpu)
        else:
            return tensor.cpu()
        
    def get_mask(self, token_sequences):
        batch_num = len(token_sequences)
        max_seq_len = max([len(word_seq) for word_seq in token_sequences])
        mask_tensor = self.to_gpu(torch.zeros(batch_num, max_seq_len, dtype=torch.float))
        for k, token_seq in enumerate(token_sequences):
            mask_tensor[k, :len(token_seq)] = 1
        return mask_tensor # batch_size x max_seq_len
        

    def forward(self, word_sequences):


        tokens_tensor, segments_tensor, number_word_in_seq = self.word_seq_indexer.batch_to_ids(word_sequences)
        self.to_gpu(tokens_tensor)
        self.to_gpu(segments_tensor)
        self.to_gpu(number_word_in_seq)
        encoded_layers = self.embeddings(tokens_tensor, segments_tensor)




        batch_embeddings = []
        for batch_i in range(tokens_tensor.shape[0]): #batch_size
            token_embeddings = []
            for token_i in range(tokens_tensor.shape[1]):  #number of token in batch element
                hidden_layers = []
                for layer_i in range(len(encoded_layers)):
                    vec = encoded_layers[layer_i][batch_i][token_i]
                    hidden_layers.append(vec)
                token_embeddings.append(hidden_layers)
            summed_last_4_layers = [torch.sum(torch.stack(layer)[-4:], 0) for layer in token_embeddings]
            summed_last_4_layers = torch.stack(summed_last_4_layers)
            batch_embeddings.append(summed_last_4_layers)

        answer = torch.stack(batch_embeddings)

        max_seq_len = max([len(word_seq) for word_seq in word_sequences])
        index =  self.to_gpu(number_word_in_seq)
        index = index.unsqueeze(2)
        index = index.repeat(1, 1, answer.shape[2])

        self_tensor = torch.zeros(index.shape) # batch_size*max_num_word (word! not token!)*len_embedding

        self_tensor = self.to_gpu(self_tensor)

        self_tensor1 = self_tensor.scatter_add_(1, index, answer)

        self_tensor1 = self_tensor1[:, 1:max_seq_len]
        self_tensor1 = F.pad(self_tensor1, (0, 0, 0, max_seq_len - self_tensor1.shape[1], 0, 0), "constant", 0)
        return self_tensor1
