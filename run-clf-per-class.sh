
GPU_1="-2"

MODEL_BERT_BASE="--path_to_bert ./pretrained/models/uncased_L-12_H-768_A-12 --embedding-dim 768"
MODEL_BERT_LARGE="--path_to_bert ./pretrained/models/uncased_bert-large --embedding-dim 1024"

DATA_TRAIN="data/targer/targer_train.txt"
DATA_TEST="data/targer/targer_test.txt"
DATA_DEV="data/targer/targer_dev.txt"
DATA="--train $DATA_TRAIN --dev $DATA_DEV --test $DATA_TEST --data-io connl-ner-2003"

PATENCE="200"
RNN_HIDDEN_DIM="1000"
EPOCHS="200"
RNN_TYPE="LSTM"
MODEL="BiRNNCRF"

TAGS=("CI" "CA" "CC" "CP")
TAGS=("CP-CI" "CP-CA" "CP-CC" "CI-CA" "CI-CP" "CA-CC")
for TAG in "${TAGS[@]}"
do
	
	FILENAME="targer-default-capri-patence-$TAG="$PATENCE"-rnn-hidden-dim="$RNN_HIDDEN_DIM"-epochs="$EPOCHS
	LOGGING="--logname $FILENAME.log --report-fn $FILENAME.txt --save $FILENAME.hdf5 --save-pred $FILENAME.out.test.json"
	DATA_TRAIN="data/targer-$TAG/targer_train.txt"
	DATA_TEST="data/targer-$TAG/targer_test.txt"
	DATA_DEV="data/targer-$TAG/targer_dev.txt"
	DATA="--train $DATA_TRAIN --dev $DATA_DEV --test $DATA_TEST --data-io connl-ner-2003"

	echo $TAG
	echo $FILENAME

	CMD="$DATA  --model $MODEL --opt sgd --save-best yes --patience $PATENCE --rnn-hidden-dim $RNN_HIDDEN_DIM --rnn-type $RNN_TYPE --gpu $GPU_1 --elmo False --epoch-num $EPOCHS --evaluator f1-alpha-match-05 --bert False --special_bert False --xlnet False $LOGGING " 
	echo "$CMD"
	echo "$FILENAME"

	python3 main.py $CMD

done
