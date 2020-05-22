

GPU_1="3"

DATA_TRAIN="data/targer/targer_train.txt"
DATA_TEST="data/targer/targer_test.txt"
DATA_DEV="data/targer/targer_dev.txt"
DATA="--train $DATA_TRAIN --dev $DATA_DEV --test $DATA_TEST --data-io connl-ner-2003"

PATENCE="100"
RNN_HIDDEN_DIM="100"
EPOCHS="10000"
LR_BERT="0.005"
MODEL_BERT_BASE="--path_to_bert xlnet-large-cased"

for i in {1..10}
do
	FILENAME="targer-xlnet-large-capri-patence="$PATENCE"-rnn-hidden-dim="$RNN_HIDDEN_DIM"-epochs="$EPOCHS"-lr_bert"$LR_BERT"-it="$i
	LOGGING="--logname $FILENAME.log --report-fn $FILENAME.txt --save $FILENAME.hdf5"
	# comparision of models

	python main.py  $DATA  --model BiRNN --embedding-dim 1024 --opt adam --save-best yes --patience $PATENCE --rnn-hidden-dim $RNN_HIDDEN_DIM --gpu $GPU_1 --epoch-num $EPOCHS --evaluator f1-macro --xlnet True $MODEL_BERT_BASE --special_bert True --lr_bert $LR_BERT $LOGGING --cross-folds-num 10 --cross-fold-id $i
done