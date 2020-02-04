

GPU_1="2"
GPU_2="3"

MODEL_BERT_BASE="--path_to_bert ./pretrained/models/uncased_L-12_H-768_A-12 --embedding-dim 768"
MODEL_BERT_LARGE="--path_to_bert ./pretrained/models/uncased_bert-large --embedding-dim 1024"

DATA_TRAIN="data/targer/targer_train.txt"
DATA_TEST="data/targer/targer_test.txt"
DATA_DEV="data/targer/targer_dev.txt"
DATA="--train $DATA_TRAIN --dev $DATA_DEV --test $DATA_TEST --data-io connl-ner-2003"

PATENCE="20"
RNN_HIDDEN_DIM="200"
EPOCHS="50"

FILENAME="targer-default-capri-patence="$PATENCE"-rnn-hidden-dim="RNN_HIDDEN_DIM"-epochs="$EPOCHS
LOGGING="--logname $FILENAME.log --report-fn $FILENAME.txt --save $FILENAME.hdf5"
# comparision of models

python main.py  $DATA  --model BiRNN --opt adam --save-best yes --patience $PATENCE --rnn-hidden-dim $RNN_HIDDEN_DIM --gpu $GPU_1 --elmo False --epoch-num $EPOCHS --evaluator f1-macro --bert False --special_bert False $LOGGING

