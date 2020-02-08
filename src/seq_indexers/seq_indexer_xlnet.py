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
    def __init__(self, gpu=-1, unique_label_list=None, check_for_lowercase=True, embeddings_dim=0, verbose=True, path_to_pretrained = None, xlnet_type = 'xlnet-base-cased', model_frozen = True):
        SeqIndexerBaseEmbeddings.__init__(self, gpu=gpu, check_for_lowercase=check_for_lowercase, zero_digits=True,
                                          pad='<pad>', unk='<unk>', load_embeddings=True, embeddings_dim=embeddings_dim,
                                          verbose=verbose, isXlnet=True)
        
        print ("create seq indexer XLNet")
        
        self.xlnet = True
        self.path_to_pretrained = path_to_pretrained
        self.tokenizer = transformers.XLNetTokenizer.from_pretrained(xlnet_type)
        self.emb = transformers.XLNetModel.from_pretrained(xlnet_type)
        self.frozen = model_frozen
        self.label_map = {label: i for i, label in enumerate(unique_label_list)}

        if gpu >= 0:
            self.emb.cuda(device=self.gpu)

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
        input_ids = batch# [x["input_ids"] for x in batch]

        MAX_LEN_IN_BATCH = np.max(np.array([len(seq) for seq in batch]))

        #word_in_sent_indexes = [x["attention_mask"] for x in batch]

        tokens_tensor = torch.tensor(input_ids)
        #word_in_sent_indexes = torch.tensor(word_in_sent_indexes)
        word_indexes = []
        for ids in input_ids:
            word_indexes.append([ids[it] if it < len(ids) else 0 for it in range(0, MAX_LEN_IN_BATCH)])
        word_indexes = torch.tensor(word_indexes)
        word_in_sent_indexes = word_indexes #torch.stack(word_indexes)
        segments_tensor = torch.tensor(np.zeros(np.mat(input_ids).shape)).to(torch.int64)
       
        return tokens_tensor, segments_tensor, word_in_sent_indexes

    def generate_input(self, words, labels, max_seq_length=64, return_input_ids=False):
        res =  self.convert_examples_to_features(words, labels, max_seq_length, self.tokenizer)
        if return_input_ids:
            return [x["input_ids"] for x in res]
        return res

    def convert_examples_to_features(self,
        tokenlist,
        labels,
        max_seq_length,
        tokenizer,
        cls_token_at_end=True,
        cls_token="<cls>",
        cls_token_segment_id=2,
        sep_token="<sep>",
        sep_token_extra=False,
        pad_on_left=True,
        pad_token=5,
        pad_token_segment_id=4,
        pad_token_label_id=0,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
    ):
        """ Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """


        features = []
        for (ex_index, _) in enumerate(tokenlist):

            tokens = []
            label_ids = []
            for word, label in zip(tokenlist[ex_index], labels[ex_index]):
                word_tokens = tokenizer.tokenize(word)
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label_ids.extend([self.label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[: (max_seq_length - special_tokens_count)]
                label_ids = label_ids[: (max_seq_length - special_tokens_count)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]
                label_ids += [pad_token_label_id]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            if cls_token_at_end:
                tokens += [cls_token]
                label_ids += [pad_token_label_id]
                segment_ids += [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                label_ids = [pad_token_label_id] + label_ids
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                label_ids = ([pad_token_label_id] * padding_length) + label_ids
            else:
                input_ids += [pad_token] * padding_length
                input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                segment_ids += [pad_token_segment_id] * padding_length
                label_ids += [pad_token_label_id] * padding_length

            #input_ids = torch.tensor(input_ids)
            #input_mask = torch.tensor(input_mask)
            #label_ids = torch.tensor(label_ids)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length

            features.append({"input_ids": input_ids, "attention_mask": input_mask, "labels": label_ids, "segment_ids": label_ids})
        return features