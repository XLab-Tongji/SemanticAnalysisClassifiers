import torchtext.data as data
from input import Dataset
from config import Config
import models
import torch
import train
import jieba.posseg as pseg
from torchtext.vocab import Vectors
import warnings
warnings.filterwarnings("ignore")

def get_data_iter(text_field, label_field, config, weights):
    train_data, test_data = Dataset.split("0", text_field, label_field, config)
    if config.PRETRAINED:
        assert weights is not None
        text_field.build_vocab([{key: 1} for key in weights.itos], vectors=weights)
    else:
        text_field.build_vocab(train_data, vectors=None)
    label_field.build_vocab(train_data)
    return data.Iterator(train_data, config.BATCH_SIZE), data.Iterator(test_data, config.BATCH_SIZE)

def chinese_tokenizer(sentence):
    exclusion = ["e", "x", "y"]  # e 叹词  x 非语素词  y 语气词
    return [word for (word, flag) in pseg.cut(sentence) if flag not in exclusion]

def main():
    config = Config()
    text_field = data.Field(tokenize=chinese_tokenizer)
    label_field = data.Field(sequential=False)


    weights = Vectors(name=config.PRETRAINED_EMBEDDING, cache=".vector_cache/") if config.PRETRAINED and config.PRETRAINED_EMBEDDING is not None else None
    train_iter, test_iter = get_data_iter(text_field, label_field, config, weights)

    config.EMBED_NUM = len(text_field.vocab)
    config.EMBED_DIM = len(weights.vectors[0]) if config.PRETRAINED else config.EMBED_DIM
    config.CLASS_NUM = len(label_field.vocab) - 1

    model = models.FastText(config, text_field.vocab.vectors)
    if config.CUDA:
        torch.cuda.set_device(config.DEVICE)
        model = model.cuda()

    if hasattr(config, "TRAIN") and config.TRAIN:
        train.train(train_iter, model, config)
    if hasattr(config, "TEST") and config.TEST:
        for i in range(5, config.EPOCHS + 1, 5):
            print("{} :".format(i), end="")
            model.load_state_dict(torch.load("ckpts/{}/snapshot_steps_{}.pt".format(type(model).__name__, i)))
            train.eval(test_iter, model, config)
    if hasattr(config, "PREDICT") and config.PREDICT:
        model.load_state_dict(torch.load("ckpts/{}/snapshot_steps_15.pt".format(type(model).__name__)))
        while True:
            print("---请输入---")
            sentence = input()
            train.predict(model, sentence, text_field, label_field, config)

main()
