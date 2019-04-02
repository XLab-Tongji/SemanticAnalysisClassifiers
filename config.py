class Config:
    # learning
    EPOCHS = 50
    LEARNING_RATE = 0.001
    BATCH_SIZE = 128
    SAVE_INTERVAL = 5

    # embedding
    PRETRAINED = True
    PRETRAINED_EMBEDDING = "embeddings/sgns.weibo.bigram"
    # PRETRAINED_EMBEDDING = "embeddings/sample"

    # models
    EMBED_NUM = None
    EMBED_DIM = 128
    DROP_OUT = 0.5
    #  cnn
    KERNEL_NUM = 100
    CNN_KERNEL_SIZES = [1, 2, 3]
    #  RNN
    RNN_HIDDEN_SIZE = 64
    RNN_HIDDEN_LAYERS = 2

    # device
    CUDA = False
    DEVICE = 0

    # option
    SHUFFLE = True
    # TRAIN = True
    # TEST = True
    PREDICT = True
