import argparse

import torch
import wget
import zipfile
import tensorflow as tf


import logging
import os

from pytorch_transformers import BertForPreTraining, BertConfig, load_tf_weights_in_bert


def add_logger(args):
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logging.info("log to stdout")
    if args.logname != None:
        log_path = os.path.join(args.model_dir)
        file_name = args.logname
        log_file = os.path.join(log_path, file_name)
        fileHandler = logging.FileHandler(log_file)
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)
        logger.info("log to {}".format(log_file))
    logger.info("===================== START ===================")
    logger.info(args.__dict__)


    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    args.logger = logger
    return logger


def download_model(args):
    file_name = args.url.split("/")[-1]
    target_file = os.path.join(args.model_dir, file_name)

    if os.path.exists(target_file) and not args.force_download or args.model_extracted_dir is not None:
        args.logger.info("file is already loaded")
    else:
        args.logger.info("save file from {} to {}".format(args.url, target_file))
        wget.download(args.url, args.model_dir + "/" + file_name)
        args.logger.info("ready")

    if args.model_extracted_dir is None or args.force_download:
        args.logger.info("extract file")
        zip_ref = zipfile.ZipFile(target_file, 'r')
        zip_ref.extractall(args.model_dir)
        zip_ref.close()
        extracted_dirname = list(zip_ref.NameToInfo.values())[0].filename

        for file in zip_ref.NameToInfo.values():
            print(file)

        target_dir = os.path.join(args.model_dir, extracted_dirname)
        logging.info("extract ready to {}".format(target_dir))
        args.model_extracted_dir = target_dir
    else:
        logging.info("I'd guess that the models are already extracted to {}".format(args.model_extracted_dir))

    return args.model_extracted_dir

def restore_model(args):
    with tf.Session() as sess:

        filelist = os.listdir(args.model_extracted_dir)

        tf_model_ckpt_meta = args.model_extracted_dir + [file for file in filelist if file.endswith(".meta")][0]
        tf_model_ckpt = args.model_extracted_dir + "bert_model.ckpt" #[file for file in filelist if file.endswith(".ckpt") or ".ckpt.index" in file][0]
        tf_model_config = args.model_extracted_dir +[file for file in filelist if file.endswith("_config.json")][0]
        pytorch_dump_path = args.model_extracted_dir + "pytorch_model.bin"
        logging.info("open checkpoint {} with meta {}".format(tf_model_ckpt, tf_model_ckpt_meta))
        saver = tf.train.import_meta_graph(tf_model_ckpt_meta)
        saver.restore(sess, tf_model_ckpt)
        config = BertConfig.from_json_file(tf_model_config)
        logging.info("Building PyTorch model from configuration: {}".format(str(config)))
        model = BertForPreTraining(config)
        # Load weights from tf checkpoint
        load_tf_weights_in_bert(model, config, tf_model_ckpt)

        # Save pytorch-model
        logging.info("Save PyTorch model to {}".format(pytorch_dump_path))
        torch.save(model.state_dict(), pytorch_dump_path)
        logging.info("saved")




def main():
    parser = argparse.ArgumentParser(description='Download tensorflow models and convert them to pytorch format')
    # logging
    parser.add_argument('--logname', type=str, default=None, help='name of file where we save a copy of the log output')

    # download and save model
    parser.add_argument('--url', type=str, default='https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip', help='url of model for download')
    parser.add_argument('--model-dir', type=str, default="./model", help="directory all models are saved")
    parser.add_argument('--model_extracted_dir', type=str, default=None, help="directory the extracted model is saved. if None we will create a new one.")
    parser.add_argument('--force-download', type=bool, default=False, help="force download of model")

    args = parser.parse_args()
    logger = add_logger(args)

    download_model(args)

    restore_model(args)


    pass

if __name__ == "__main__":
    main()