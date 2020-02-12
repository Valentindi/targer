

GPU_1="3"

DATA_TRAIN="data/targer/targer_train.txt"
DATA_TEST="data/targer/targer_test.txt"
DATA_DEV="data/targer/targer_dev.txt"
DATA="--train $DATA_TRAIN --dev $DATA_DEV --test $DATA_TEST --data-io connl-ner-2003"

PATENCE="100"
RNN_HIDDEN_DIM="500"
EPOCHS="10000"
LR_BERT="0.005"
FROZEN="TRUE"
TARGER_MODEL="BiRNN"
MODEL_BERT_BASE="--path_to_bert xlnet-base-cased"
FILENAME="targer-xlnet-base-capri-patence="$PATENCE"-rnn-hidden-dim="$RNN_HIDDEN_DIM"-epochs="$EPOCHS"-lr_bert"$LR_BERT"-frozen="$FROZEN"-lr"$LR-model"TARGER_MODEL"
LOGGING="--logname $FILENAME.log --report-fn $FILENAME.txt --save $FILENAME.hdf5"
# comparision of models

python main.py  $DATA --seed-num 42 --model BiRNN --embedding-dim 768 --opt adam --save-best yes --patience $PATENCE --rnn-hidden-dim $RNN_HIDDEN_DIM --gpu $GPU_1 --epoch-num $EPOCHS --evaluator f1-macro --xlnet True $MODEL_BERT_BASE --special_bert True --lr_bert $LR_BERT $LOGGING --bert_frozen $FROZEN --model $TARGER_MODEL
