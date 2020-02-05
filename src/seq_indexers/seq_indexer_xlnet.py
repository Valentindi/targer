"""indexer for XLNet model"""
import transformers as transformers

"""join list of input words into string"""
"""provide XLNet tokenization"""

from src.seq_indexers.seq_indexer_base_embeddings import SeqIndexerBaseEmbeddings
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import torch

class SeqIndexerXlnet(SeqIndexerBaseEmbeddings):
    """SeqIndexerWord converts list of lists of words as strings to list of lists of integer indices and back."""
    def __init__(self, gpu=-1, check_for_lowercase=True, embeddings_dim=0, verbose=True, path_to_pretrained = None, xlnet_type = 'xlnet-base-cased', model_frozen = True):
        SeqIndexerBaseEmbeddings.__init__(self, gpu=gpu, check_for_lowercase=check_for_lowercase, zero_digits=True,
                                          pad='<pad>', unk='<unk>', load_embeddings=True, embeddings_dim=embeddings_dim,
                                          verbose=verbose, isXlnet=True)
        
        print ("create seq indexer XLNet")
        
        self.xlnet = True
        self.path_to_pretrained = path_to_pretrained
        self.tokenizer = transformers.XLNetTokenizer.from_pretrained(xlnet_type)
        self.emb = transformers.XLNetModel.from_pretrained(xlnet_type)
        self.frozen = model_frozen

        input_ids = torch.tensor([self.tokenizer.encode("Here is some text to encode", add_special_tokens=True)])

        for param in self.emb.parameters():
            param.requires_grad = False

        ## froze - unfroze layer of loaded xlnet pre-trained model. Now only pooler leayer is unfrozen. You can unfroze layers from encoders, decoders, etc.
        if (not self.frozen):
            #print ("loaded XLNET model will be trained")
            #for i in [0]:
                #for param in self.emb.encoder.layer[i].parameters():
                    #param.requires_grad = True
            for param in self.emb.pooler.parameters():
                param.requires_grad = True
        self.emb.eval()
        print ("XLnet model loaded succesifully")
        
    def get_word_indexes(self, tokenized_text, maxlen): #assign token to num of word in sentence
        i = 0
        word_index = []
        next_word = False
        for token in tokenized_text:
            if ('##' in token):
                next_word = False
            else:
                next_word = True
            if (next_word):
                i += 1
            word_index.append(i - 1)
        i += 1
        while (len(word_index) < maxlen):
            word_index.append(i - 1)
        return torch.tensor(word_index)
    

        
    def batch_to_ids(self, batch):
        batch = [' '.join(sent) for sent in batch]
        batch = ["[CLS] " + text + " [SEP]" for text in batch]
        tokenized_texts = [self.tokenizer.tokenize(sent) for sent in batch]
        MAX_LEN = np.max(np.array([len(seq) for seq in tokenized_texts]))
        MAX_LEN_IN_BATCH = np.max(np.array([len(seq) for seq in batch]))
        
        word_indexes = [self.get_word_indexes(text, MAX_LEN) for text in tokenized_texts]
        
        input_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts], maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
        word_in_sent_indexes = torch.stack(word_indexes)
        
        tokens_tensor = torch.tensor(input_ids)
        segments_tensor = torch.tensor(np.zeros(input_ids.shape)).to(torch.int64)
       
        return tokens_tensor, segments_tensor, word_in_sent_indexes
        