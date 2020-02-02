python -m spacy download en_core_web_sm
python -m spacy download en
export PYTHONIOENCODING=utf8
cd pretrained
mkdir models
python load_model.py --model-dir ./models/ --logname load_model.log

mkdir bert-large-uncased
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin -O pytorch_model.bin
wget wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt -O vocab.txt
 wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-config.json -O bert_config.json
