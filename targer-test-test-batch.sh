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

name="run-vanilla"
python3 main.py --train $train --dev $dev --test $test --save $name.hdf5 --rnn-type=Vanilla --gpu -1 --save-best yes --cross-folds-num 5 -v $eval -p 25 -e $e

mv 2019* $targetdir
mv name.hdf5 $targetdir
$(cd $targetdir && git add .)
$(cd $targetdir && git commit -m "new updates from targer ($name)")
$(cd $targetdir && git push)



cp -f ../glove.6B.100d.txt embeddings/glove.6B.100d.txt

name="run-lstm"
python3 main.py --train $train --dev $dev --test $test --save $name.hdf5 --rnn-type=LSTM --gpu -1 --save-best yes --cross-folds-num 5 -v $eval -p 25 -e $e

mv 2019* $targetdir
mv name.hdf5 $targetdir
$(cd $targetdir && git add .)
$(cd $targetdir && git commit -m "new updates from targer ($name)")
$(cd $targetdir && git push)


cp -f ../glove.6B.100d.txt embeddings/glove.6B.100d.txt

name="run-GRU"
python3 main.py --train $train --dev $dev --test $test --save $name.hdf5 --rnn-type=GRU --gpu -1 --save-best yes --cross-folds-num 5 -v $eval -p 25 -e $e

mv 2019* $targetdir
mv name.hdf5 $targetdir
$(cd $targetdir && git add .)
$(cd $targetdir && git commit -m "new updates from targer ($name)")
$(cd $targetdir && git push)


cp -f ../glove.6B.100d.txt embeddings/glove.6B.100d.txt

name="opt-sgd"
python3 main.py --train $train --dev $dev --test $test --save $name.hdf5 --opt=sgd --gpu -1 --save-best yes --cross-folds-num 5 -v $eval -p 25 -e $e

mv 2019* $targetdir
mv name.hdf5 $targetdir
$(cd $targetdir && git add .)
$(cd $targetdir && git commit -m "new updates from targer ($name)")
$(cd $targetdir && git push)


cp -f ../glove.6B.100d.txt embeddings/glove.6B.100d.txt

name="opt-adam"
python3 main.py --train $train --dev $dev --test $test --save $name.hdf5 --opt=adam --gpu -1 --save-best yes --cross-folds-num 5 -v $eval -p 25 -e $e

mv 2019* $targetdir
mv name.hdf5 $targetdir
$(cd $targetdir && git add .)
$(cd $targetdir && git commit -m "new updates from targer ($name)")
$(cd $targetdir && git push)


