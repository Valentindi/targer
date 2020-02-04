

GPU_1="5"

MODEL_BERT_BASE="--path_to_bert ./pretrained/models/uncased_L-12_H-768_A-12 --embedding-dim 1024"
MODEL_BERT_LARGE="--path_to_bert ./pretrained/models/bert-large-uncased --embedding-dim 1024"

DATA_TRAIN="data/NER/CoNNL_2003_shared_task/train.txt"
DATA_TEST="data/NER/CoNNL_2003_shared_task/test.txt"
DATA_DEV="data/NER/CoNNL_2003_shared_task/dev.txt"
DATA="--train $DATA_TRAIN --dev $DATA_DEV --test $DATA_TEST --data-io connl-ner-2003"

PATENCE="200"
RNN_HIDDEN_DIM="200"
EPOCHS="200"

FILENAME="targer-bert-large-nar-patence="$PATENCE"-rnn-hidden-dim="$RNN_HIDDEN_DIM"-epochs="$EPOCHS
LOGGING="--logname $FILENAME.log --report-fn $FILENAME.txt --save $FILENAME.hdf5"
# comparision of models

#python main.py  $DATA  --model BiRNN --opt adam --save-best yes --patience $PATENCE --rnn-hidden-dim $RNN_HIDDEN_DIM --gpu $GPU_1 --elmo False --epoch-num 50 --evaluator f1-macro --bert True $MODEL_BERT_BASE --logname bert-base.log --report-fn bert-base.txt &
python main.py $DATA --model BiRNN --opt adam --save-best yes --patience $PATENCE --rnn-hidden-dim $RNN_HIDDEN_DIM --gpu $GPU_2 --elmo False --epoch-num $EPOCHS --evaluator f1-macro --bert True  $MODEL_BERT_LARGE $LOGGING