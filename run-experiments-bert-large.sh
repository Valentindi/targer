

GPU_1="2"

MODEL_BERT_BASE="--path_to_bert ./pretrained/models/uncased_L-12_H-768_A-12 --embedding-dim 1024"
MODEL_BERT_LARGE="--path_to_bert ./pretrained/models/bert-large-uncased --embedding-dim 1024"

DATA_TRAIN="data/targer/targer_train.txt"
DATA_TEST="data/targer/targer_test.txt"
DATA_DEV="data/targer/targer_dev.txt"
DATA="--train $DATA_TRAIN --dev $DATA_DEV --test $DATA_TEST --data-io connl-ner-2003"

PATENCE="200"
RNN_HIDDEN_DIM="200"
EPOCHS="200"

FILENAME="targer-bert-large-capri-patence="$PATENCE"-rnn-hidden-dim="$RNN_HIDDEN_DIM"-epochs="$EPOCHS
LOGGING="--logname $FILENAME.log --report-fn $FILENAME.txt --save $FILENAME.hdf5"
# comparision of models

for i in {1..10}
do
	FILENAME="targer-bert-large-capri-patence="$PATENCE"-rnn-hidden-dim="$RNN_HIDDEN_DIM"-epochs="$EPOCHS"-it="$i
	LOGGING="--logname $FILENAME.log --report-fn $FILENAME.txt --save $FILENAME.hdf5"
python main.py $DATA --model BiRNN --opt adam --save-best yes --patience $PATENCE --rnn-hidden-dim $RNN_HIDDEN_DIM --gpu $GPU_2 --elmo False --epoch-num $EPOCHS --evaluator f1-macro --bert True  $MODEL_BERT_LARGE --special_bert True $LOGGING --cross-folds-num 10 --cross-fold-id $i

done