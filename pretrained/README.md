# This file contains all you need to work with pretrained models
````
├── README.md
├── file_utils_custom.py
├── load_model.py # script for download and convert tensorflow model to pytorch model
├── tagger_NER.hdf5 # pretrained model
├── test_model
│   ├── test.log # log output
│   ├── uncased_L-12_H-768_A-12 # extracted model
│   └── uncased_L-12_H-768_A-12.zip # zipped model
└── tokenizer_custom_bert.py
````

## Load and convert tensorflow-model
````
Download tensorflow models and convert them to pytorch format

optional arguments:
  -h, --help            show this help message and exit
  --logname LOGNAME     name of file where we save a copy of the log output
  --url URL             url of model for download
  --model-dir MODEL_DIR
                        directory all models are saved
  --model_extracted_dir MODEL_EXTRACTED_DIR
                        directory the extracted model is saved. if None we
                        will create a new one.
  --force-download FORCE_DOWNLOAD
                        force download of model

````