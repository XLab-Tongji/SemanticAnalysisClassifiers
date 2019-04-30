import os
import torch
import torch.nn.functional as F
import datetime
import torch.autograd as autograd

def train(train_iter, model, config, text_field):
    if config.CUDA:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    count, loss_sum = 0, 0
    model.train()
    for epoch in range(1, config.EPOCHS + 1):
        print("Epoch_{}_{}".format(epoch, datetime.datetime.now().strftime('%H-%M-%S')))
        for i, batch in enumerate(train_iter):
            # 调用text_field.vocab的itos、stoi函数 把batch.text中识别为0（unk）的字符找到，并定位该句子
            features, target = batch.text, batch.label
            features.t_(), target.sub_(1)
            if config.CUDA:
                features, target = features.cuda(), target.cuda()
            for j, sentence in enumerate(features):
                for k, token in enumerate(sentence):
                    if token == 0:
                        print(batch.dataset.examples[i * config.BATCH_SIZE + j].text[k])

            # for sentence in feature:
            #       for token in sentence:
            #           if token == 0:
            #               then identify the id of sentence, token
            optimizer.zero_grad()
            log = model(features)
            loss = F.cross_entropy(log, target)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            count += 1
            print_num = 100
            if count == print_num:
                print(" loss %.5f" % (loss_sum / print_num))
                count, loss_sum = 0, 0

        if epoch % config.SAVE_INTERVAL == 0:
            save(model, epoch, "snapshot")

def eval(test_iter, model, config):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in test_iter:
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if config.CUDA:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.item()
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(test_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(avg_loss, accuracy, corrects, size))
    return accuracy

def predict(model, sentence, text_field, label_field, config):
    model.eval()
    sentence = text_field.preprocess(sentence)
    while len(sentence) < 3:
        sentence.append("<pad>")
    sentence = [[text_field.vocab.stoi[x] for x in sentence]]
    x = torch.tensor(sentence)
    x = autograd.Variable(x)
    if config.CUDA:
        x = x.cuda()
    output = model(x)
    _, pred = torch.max(output, 1)
    print(label_field.vocab.itos[pred.data[0]+1])

def save(model, num, prefix):
    dirName = 'ckpts' + '/' + type(model).__name__
    if not os.path.isdir(dirName):
        os.makedirs(dirName)
    prefix = os.path.join(dirName, prefix)
    path = "{}_steps_{}.pt".format(prefix, num)
    torch.save(model.state_dict(), path)
