python -m spacy download en_core_web_sm
python -m spacy download en
export PYTHONIOENCODING=utf8
python pretrained/load_model.py --model-dir ./pretrained/models/ --logname load_model.log
python main.py --train data/targer/targer_train.txt --dev data/targer/targer_dev.txt --test data/targer/targer_test.txt  --data-io connl-ner-2003 --model BiRNN --opt adam --save-best yes --patience 20 --rnn-hidden-dim 200 --gpu -1 --elmo False --epoch-num 50 --evaluator f1-macro --bert True --path_to_bert ./pretrained/uncased_L-12_H-768_A-12/pytorch_model --logname bert-768

