python3 -m spacy download en_core_web_sm
python3 -m spacy download en
python3 main.py --train data/targer/targer_train.txt --dev data/targer/targer_dev.txt --test data/targer/targer_test.txt  --data-io connl-ner-2003 --model BiRNN --opt adam --save-best yes --patience 20 --rnn-hidden-dim 200 --gpu -1 --elmo False --epoch-num 50 --evaluator f1-macro --bert False