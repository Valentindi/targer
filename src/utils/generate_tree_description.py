import os
import logging

def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = '|__' * (level)
        dir_name = os.path.basename(root)
        if dir_name.startswith('__'):
            continue
        if dir_name.startswith('.'):
            continue
        logging.info('{}{}/'.format(indent, dir_name))
        subindent = '   ' * (level) + '|__'
        for short_fn in files:
            fn = os.path.join(root, short_fn)
            ext = short_fn.split('.')[-1]
            if ext == 'py':
                logging.info('{}{} --> {}'.format(subindent, short_fn, read_description(fn)))
            else:
                logging.info('{}{} --> '.format(subindent, short_fn))


def read_description(fn):
    with open(fn) as f:
        first_line = f.readline()
    return first_line.replace('"""', '').replace('\n', '')

if __name__ == "__main__":
    main_path = os.path.join(os.path.dirname(__file__), '../')
    os.chdir(main_path)
    logging.info('|__ articles/ --> collection of papers related to the tagging, argument mining, etc.')
    logging.info('|__ data/')
    logging.info('        |__ NER/ --> Datasets for Named Entity Recognition')
    logging.info('            |__ CoNNL_2003_shared_task/ --> data for NER CoNLL-2003 shared task (English) in BOI-2')
    logging.info('                                            CoNNL format, from E.F. Tjong Kim Sang and F. De Meulder,')
    logging.info('                                            Introduction to the CoNLL-2003 Shared Task:')
    logging.info('                                            Language-Independent Named Entity Recognition, 2003.')
    logging.info('        |__ AM/ --> Datasets for Argument Mining')
    logging.info('            |__ persuasive_essays/ --> data for persuasive essays in BOI-2-like CoNNL format, from:')
    logging.info('                                       Steffen Eger, Johannes Daxenberger, Iryna Gurevych. Neural')
    logging.info('                                       End-to-End  Learning for Computational Argumentation Mining, 2017')
    logging.info('|__ docs/ --> documentation')
    logging.info('|__ embeddings')
    logging.info('        |__ get_glove_embeddings.sh --> script for downloading GloVe6B 100-dimensional word embeddings')
    logging.info('        |__ get_fasttext_embeddings.sh --> script for downloading Fasttext word embeddings')
    logging.info('|__ pretrained/')
    logging.info('        |__ tagger_NER.hdf5 --> tagger for NER, BiLSTM+CNN+CRF trained on NER-2003 shared task, English')
    list_files(startpath=os.getcwd())
    logging.info('|__ main.py --> main script for training/evaluation/saving tagger models')
    logging.info('|__ run_tagger.py --> run the trained tagger model from the checkpoint file')
    logging.info('|__ conlleval --> "official" Perl script from NER 2003 shared task for evaluating the f1 scores,'
        '\n                   author: Erik Tjong Kim Sang, version: 2004-01-26')
    logging.info('|__ requirements.txt --> file for managing packages requirements')
