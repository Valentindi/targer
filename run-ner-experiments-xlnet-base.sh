

GPU_1="1"

DATA_TRAIN="data/NER/CoNNL_2003_shared_task/train.txt"
DATA_TEST="data/NER/CoNNL_2003_shared_task/test.txt"
DATA_DEV="data/NER/CoNNL_2003_shared_task/dev.txt"
DATA="--train $DATA_TRAIN --dev $DATA_DEV --test $DATA_TEST --data-io connl-ner-2003"

PATENCE="200"
RNN_HIDDEN_DIM="200"
EPOCHS="200"

FILENAME="targer-xlnet-large-capri-patence="$PATENCE"-rnn-hidden-dim="$RNN_HIDDEN_DIM"-epochs="$EPOCHS
LOGGING="--logname $FILENAME.log --report-fn $FILENAME.txt --save $FILENAME.hdf5"
# comparision of models

python main.py  $DATA  --model BiRNN --embedding-dim 768 --opt adam --save-best yes --patience $PATENCE --rnn-hidden-dim $RNN_HIDDEN_DIM --gpu $GPU_1 --epoch-num $EPOCHS --evaluator f1-macro --xlnet True $MODEL_BERT_BASE --special_bert True $LOGGING
