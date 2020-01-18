export PYTHONIOENCODING=utf8

train="../wstud-thesis-dittmar/data/targer/targer_train.txt"
dev="../wstud-thesis-dittmar/data/targer/targer_dev.txt"
test="../wstud-thesis-dittmar/data/targer/targer_test.txt"
targetdir="../wstud-thesis-dittmar/targer-results"

e=1000

eval='f1-alpha-match-05'

# TODOs:  
#	--model {BiRNN,BiRNNCNN,BiRNNCRF,BiRNNCNNCRF}
# 	--opt {sgd,adam} x
# 	--LR {0.01, 0.001, 0.1, 1}
# 	--rnn-type {Vanilla,LSTM,GRU} x
# 	--rnn-hidden-dim RNN_HIDDEN_DIM {10, 100, 1000}

cp -f ../glove.6B.100d.txt embeddings/glove.6B.100d.txt

name="run-hidden-200"
python main.py --train $train --dev $dev --test $test --save $name.hdf5 --rnn-hidden-dim 200  --save-best yes --cross-folds-num 5 -v $eval -p 25 -e $e
python run_tagger.py $name.hdf5 $test -g -1 -o $name.json
python eval.py $test $name.json $name.eval.txt
python generate_detailed_report.py $test $name.json $name.report.html

mv $name* targetdir



cp -f ../glove.6B.100d.txt embeddings/glove.6B.100d.txt

name="run-hidden-300"
python main.py --train $train --dev $dev --test $test --save $name.hdf5 --rnn-hidden-dim 300  --save-best yes --cross-folds-num 5 -v $eval -p 25 -e $e
python run_tagger.py $name.hdf5 $test -g -1 -o $name.json
python eval.py $test $name.json $name.eval.txt
python generate_detailed_report.py $test $name.json $name.report.html

mv $name* targetdir




cp -f ../glove.6B.100d.txt embeddings/glove.6B.100d.txt

name="run-hidden-400"
python main.py --train $train --dev $dev --test $test --save $name.hdf5 --rnn-hidden-dim 400  --save-best yes --cross-folds-num 5 -v $eval -p 25 -e $e
python run_tagger.py $name.hdf5 $test -g -1 -o $name.json
python eval.py $test $name.json $name.eval.txt
python generate_detailed_report.py $test $name.json $name.report.html

mv $name* targetdir




cp -f ../glove.6B.100d.txt embeddings/glove.6B.100d.txt

name="run-hidden-666"
python main.py --train $train --dev $dev --test $test --save $name.hdf5 --rnn-hidden-dim 666  --save-best yes --cross-folds-num 5 -v $eval -p 25 -e $e
python run_tagger.py $name.hdf5 $test -g -1 -o $name.json
python eval.py $test $name.json $name.eval.txt
python generate_detailed_report.py $test $name.json $name.report.html

mv $name* targetdir



cp -f ../glove.6B.100d.txt embeddings/glove.6B.100d.txt

name="lr-0.001"
python main.py --train $train --dev $dev --test $test --save $name.hdf5 --lr 0.001  --save-best yes --cross-folds-num 5 -v $eval -p 25 -e $e
python run_tagger.py $name.hdf5 $test -g -1 -o $name.json
python eval.py $test $name.json $name.eval.txt
python generate_detailed_report.py $test $name.json $name.report.html

mv $name* targetdir




cp -f ../glove.6B.100d.txt embeddings/glove.6B.100d.txt

name="lr-0.005"
python main.py --train $train --dev $dev --test $test --save $name.hdf5 --lr 0.005  --save-best yes --cross-folds-num 5 -v $eval -p 25 -e $e
python run_tagger.py $name.hdf5 $test -g -1 -o $name.json
python eval.py $test $name.json $name.eval.txt
python generate_detailed_report.py $test $name.json $name.report.html

mv $name* targetdir



cp -f ../glove.6B.100d.txt embeddings/glove.6B.100d.txt

name="lr-0.05"
python main.py --train $train --dev $dev --test $test --save $name.hdf5 --lr 0.05  --save-best yes --cross-folds-num 5 -v $eval -p 25 -e $e
python run_tagger.py $name.hdf5 $test -g -1 -o $name.json
python eval.py $test $name.json $name.eval.txt
python generate_detailed_report.py $test $name.json $name.report.html

mv $name* targetdir



name="run-BiRNNCRF-hidden-500-GRO"
python main.py --train $train --dev $dev --test $test --save $name.hdf5  --model BiRNNCRF --rnn-type=GRU --rnn-hidden-dim 500   --save-best yes --cross-folds-num 5 -v $eval -p 25 -e $e
python run_tagger.py $name.hdf5 $test -g -1 -o $name.json
python eval.py $test $name.json $name.eval.txt
python generate_detailed_report.py $test $name.json $name.report.html

mv $name* targetdir








