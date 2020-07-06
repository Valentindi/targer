
GPU_1="2"

MODEL_BERT_BASE="--path_to_bert ./pretrained/models/uncased_L-12_H-768_A-12 --embedding-dim 768"
MODEL_BERT_LARGE="--path_to_bert ./pretrained/models/uncased_bert-large --embedding-dim 1024"

DATA_TRAIN="data/targer/targer_train.txt"
DATA_TEST="data/targer/targer_test.txt"
DATA_DEV="data/targer/targer_dev.txt"
DATA="--train $DATA_TRAIN --dev $DATA_DEV --test $DATA_TEST --data-io connl-ner-2003"

PATENCE="200"
RNN_HIDDEN_DIM="500"
EPOCHS="200"
RNN_TYPE="GRU"
MODEL="BiRNNCRF"


for i in {1..10}
do

	FILENAME="targer-default-capri-patence="$PATENCE"-rnn-hidden-dim="$RNN_HIDDEN_DIM"-epochs="$EPOCHS"-it="$i
	LOGGING="--logname $FILENAME.log --report-fn $FILENAME.txt --save $FILENAME.hdf5"
	# comparision of models

	CMD="$DATA  --model $MODEL --opt adam --save-best yes --patience $PATENCE --rnn-hidden-dim $RNN_HIDDEN_DIM --rnn-type $RNN_TYPE --gpu $GPU_1 --elmo False --epoch-num $EPOCHS --evaluator f1-macro --bert False --special_bert False --xlnet False $LOGGING --cross-folds-num 10 --cross-fold-id $i" 
	python run_tagger.py "$FILENAME.hdf5" $DATA_DEV --output "$FILENAME.out.json" -v f1-macro --gpu $GPU_1 

	echo "$CMD"
	echo "$FILENAME"

	python3 main.py $CMD

done
