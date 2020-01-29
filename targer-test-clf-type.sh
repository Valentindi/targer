export PYTHONIOENCODING=utf8

train="../wstud-thesis-dittmar/data/targer/targer_train.txt"
dev="../wstud-thesis-dittmar/data/targer/targer_dev.txt"
test="../wstud-thesis-dittmar/data/targer/targer_test.txt"
targetdir="../wstud-thesis-dittmar/targer-results"

e=250

eval='f1-alpha-match-05'

# TODOs:  
#	--model {BiRNN,BiRNNCNN,BiRNNCRF,BiRNNCNNCRF}
# 	--opt {sgd,adam} x
# 	--LR {0.01, 0.001, 0.1, 1}
# 	--rnn-type {Vanilla,LSTM,GRU} x
# 	--rnn-hidden-dim RNN_HIDDEN_DIM {10, 100, 1000}

cp -f ../glove.6B.100d.txt embeddings/glove.6B.100d.txt

name="run-BiRNN"
python3 main.py --train $train --dev $dev --test $test --save $name.hdf5 --model BiRNN --gpu -1 --save-best yes --cross-folds-num 5 -v $eval -p 25 -e $e
python3 run_tagger.py $name.hdf5 $test -g -1 -o $name.json
python3 eval.py $test $name.json $name.eval.txt
python3 generate_detailed_report.py $test $name.json $name.report.html

mv $name* targetdir
$(cd $targetdir && git add .)
$(cd $targetdir && git commit -m "new updates from targer ($name)")
$(cd $targetdir && git push)


cp -f ../glove.6B.100d.txt embeddings/glove.6B.100d.txt

name="run-BiRNNCNN"
python3 main.py --train $train --dev $dev --test $test --save $name.hdf5 --model BiRNNCNN --gpu -1 --save-best yes --cross-folds-num 5 -v $eval -p 25 -e $e
python3 run_tagger.py $name.hdf5 $test -g -1 -o $name.json
python3 eval.py $test $name.json $name.eval.txt
python3 generate_detailed_report.py $test $name.json $name.report.html

mv $name* targetdir
$(cd $targetdir && git add .)
$(cd $targetdir && git commit -m "new updates from targer ($name)")
$(cd $targetdir && git push)


cp -f ../glove.6B.100d.txt embeddings/glove.6B.100d.txt

name="run-BiRNNCRF"
python3 main.py --train $train --dev $dev --test $test --save $name.hdf5 --model BiRNNCRF --gpu -1 --save-best yes --cross-folds-num 5 -v $eval -p 25 -e $e
python3 run_tagger.py $name.hdf5 $test -g -1 -o $name.json
python3 eval.py $test $name.json $name.eval.txt
python3 generate_detailed_report.py $test $name.json $name.report.html

mv $name* targetdir
$(cd $targetdir && git add .)
$(cd $targetdir && git commit -m "new updates from targer ($name)")
$(cd $targetdir && git push)


cp -f ../glove.6B.100d.txt embeddings/glove.6B.100d.txt

name="run-BiRNNCNNCRF"
python3 main.py --train $train --dev $dev --test $test --save $name.hdf5 --model BiRNNCNNCRF --gpu -1 --save-best yes --cross-folds-num 5 -v $eval -p 25 -e $e
python3 run_tagger.py $name.hdf5 $test -g -1 -o $name.json
python3 eval.py $test $name.json $name.eval.txt
python3 generate_detailed_report.py $test $name.json $name.report.html

mv $name* targetdir
$(cd $targetdir && git add .)
$(cd $targetdir && git commit -m "new updates from targer ($name)")
$(cd $targetdir && git push)






