
GPU_1="-1"

MODEL_BERT_BASE="--path_to_bert ./pretrained/models/uncased_L-12_H-768_A-12 --embedding-dim 1024"
MODEL_BERT_LARGE="--path_to_bert ./pretrained/models/bert-large-uncased --embedding-dim 1024"

DATA_TRAIN="data/targer/targer_train.txt"
DATA_TEST="data/targer/targer_test.txt"
DATA_DEV="data/targer/targer_dev.txt"
DATA="--train $DATA_TRAIN --dev $DATA_DEV --test $DATA_TEST --data-io connl-ner-2003"

PATENCE="1"
RNN_HIDDEN_DIM="1"
EPOCHS="1"

#FILENAME="targer-bert-large-capri-patence="$PATENCE"-rnn-hidden-dim="$RNN_HIDDEN_DIM"-epochs="$EPOCHS
#LOGGING="--logname $FILENAME.log --report-fn $FILENAME.txt --save $FILENAME.hdf5"
#BERT="--bert True --path_to_bert ./pretrained/models/bert-base-uncased"
CMD="--train data/targer/targer_train.txt --dev data/targer/targer_dev.txt --test data/targer/targer_test.txt  --model BiRNN --opt adam --save-best yes --patience 1 --rnn-hidden-dim 1 --gpu 2 --elmo False --epoch-num 1 --evaluator f1-macro --logname bert-base --embedding-dim 768  $LOGGING  $BERT"



echo $FILENAME
FILENAME="targer-bert-large-capri-patence=$PATENCE-rnn-hidden-dim=$RNN_HIDDEN_DIM-epochs=$EPOCHS-0"
LOGGING="--logname $FILENAME.log --report-fn $FILENAME.txt --save $FILENAME.hdf5"
echo "$FILENAME"
python3 main.py $CMD --cross-folds-num 10 --cross-fold-id 0


FILENAME="targer-bert-large-capri-patence=$PATENCE-rnn-hidden-dim=$RNN_HIDDEN_DIM-epochs=$EPOCHS-1"
LOGGING="--logname $FILENAME.log --report-fn $FILENAME.txt --save $FILENAME.hdf5"
python3 main.py $CMD --cross-folds-num 10 --cross-fold-id 1


FILENAME="targer-bert-large-capri-patence=$PATENCE-rnn-hidden-dim=$RNN_HIDDEN_DIM-epochs=$EPOCHS-2"
LOGGING="--logname $FILENAME.log --report-fn $FILENAME.txt --save $FILENAME.hdf5"
python3 main.py $CMD --cross-folds-num 10 --cross-fold-id 2


FILENAME="targer-bert-large-capri-patence=$PATENCE-rnn-hidden-dim=$RNN_HIDDEN_DIM-epochs=$EPOCHS-3"
LOGGING="--logname $FILENAME.log --report-fn $FILENAME.txt --save $FILENAME.hdf5"
python3 main.py $CMD --cross-folds-num 10 --cross-fold-id 3


FILENAME="targer-bert-large-capri-patence=$PATENCE-rnn-hidden-dim=$RNN_HIDDEN_DIM-epochs=$EPOCHS-4"
LOGGING="--logname $FILENAME.log --report-fn $FILENAME.txt --save $FILENAME.hdf5"
python3 main.py $CMD --cross-folds-num 10 --cross-fold-id 4


FILENAME="targer-bert-large-capri-patence=$PATENCE-rnn-hidden-dim=$RNN_HIDDEN_DIM-epochs=$EPOCHS-5"
LOGGING="--logname $FILENAME.log --report-fn $FILENAME.txt --save $FILENAME.hdf5"
python3 main.py $CMD --cross-folds-num 10 --cross-fold-id 5


FILENAME="targer-bert-large-capri-patence=$PATENCE-rnn-hidden-dim=$RNN_HIDDEN_DIM-epochs=$EPOCHS-6"
LOGGING="--logname $FILENAME.log --report-fn $FILENAME.txt --save $FILENAME.hdf5"
python3 main.py $CMD --cross-folds-num 10 --cross-fold-id 6


FILENAME="targer-bert-large-capri-patence=$PATENCE-rnn-hidden-dim=$RNN_HIDDEN_DIM-epochs=$EPOCHS-7"
LOGGING="--logname $FILENAME.log --report-fn $FILENAME.txt --save $FILENAME.hdf5"
python3 main.py $CMD --cross-folds-num 10 --cross-fold-id 7


FILENAME="targer-bert-large-capri-patence=$PATENCE-rnn-hidden-dim=$RNN_HIDDEN_DIM-epochs=$EPOCHS-8"
LOGGING="--logname $FILENAME.log --report-fn $FILENAME.txt --save $FILENAME.hdf5"
python3 main.py $CMD --cross-folds-num 10 --cross-fold-id 8


FILENAME="targer-bert-large-capri-patence=$PATENCE-rnn-hidden-dim="$RNN_HIDDEN_DIM"-epochs="$EPOCHS-9"
LOGGING="--logname $FILENAME.log --report-fn $FILENAME.txt --save $FILENAME.hdf5"
python3 main.py $CMD --cross-folds-num 10 --cross-fold-id 9
