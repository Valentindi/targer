

GPU_1="4"

DATA_TRAIN="data/targer/targer_train.txt"
DATA_TEST="data/targer/targer_test.txt"
DATA_DEV="data/targer/targer_dev.txt"
DATA="--train $DATA_TRAIN --dev $DATA_DEV --test $DATA_TEST --data-io connl-ner-2003"

PATENCE="200"
RNN_HIDDEN_DIM="200"
EPOCHS="200"
LR_BERT="0.0005"

FILENAME="targer-xlnet-base-capri-patence="$PATENCE"-rnn-hidden-dim="$RNN_HIDDEN_DIM"-epochs="$EPOCHS"-lr_bert"$LR_BERT
LOGGING="--logname $FILENAME.log --report-fn $FILENAME.txt --save $FILENAME.hdf5"
# comparision of models

python main.py  $DATA  --model BiRNN --embedding-dim 768 --opt adam --save-best yes --patience $PATENCE --rnn-hidden-dim $RNN_HIDDEN_DIM --gpu $GPU_1 --epoch-num $EPOCHS --evaluator f1-macro --xlnet True $MODEL_BERT_BASE --special_bert True --lr_bert $LR_BERT $LOGGING
