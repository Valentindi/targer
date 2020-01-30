python -m spacy download en_core_web_sm
python -m spacy download en
export PYTHONIOENCODING=utf8
cd pretrained
mkdir models
python load_model.py --model-dir ./models/ --logname load_model.log
cd ..
python main.py --train data/targer/targer_train.txt --dev data/targer/targer_dev.txt --test data/targer/targer_test.txt  --data-io connl-ner-2003 --model BiRNN --opt adam --save-best yes --patience 20 --rnn-hidden-dim 200 --gpu 2 --elmo False --epoch-num 50 --evaluator f1-macro --bert True --path_to_bert ./pretrained/models/uncased_L-12_H-768_A-12/ --logname bert-768

