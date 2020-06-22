import argparse
import data_util
import time
import torch
import torch.nn.functional as F
import numpy as np

parser = argparse.ArgumentParser(description="Text Classification")

parser.add_argument("--task_size", type=int, default=1)
parser.add_argument("--model", choices=["LSTM", "LSTM_Attn", "CNN", "RCNN", "RNN", "SelfAttention"], default="CNN",
                    nargs="?")
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--embed_size", type=int, default=300)
parser.add_argument("--embedding", choices=["baidu", "wiki", "merge"], default="merge", nargs="?")
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--use_cuda", type=bool, default=True)

# For CNN
# The hyperparameter author used: (Ref: https://github.com/prakashpandey9/Text-Classification-Pytorch/issues/1)
# in_channels=1
# out_channels=128
# kernel_heights=[3, 4, 5]
# stride=1
# padding=0
# keep_probab=0.8
parser.add_argument("--in_channels", type=int, default=1)
parser.add_argument("--out_channels", type=int, default=128)
parser.add_argument("--kernel_heights", type=list, default=[3, 4, 5])
parser.add_argument("--stride", type=int, default=1)
parser.add_argument("--padding", type=int, default=0)
parser.add_argument("--keep_probab", type=float, default=0.8)

# For RNN and its variants
parser.add_argument("--hidden_size", type=int, default=256)

# For Testing
parser.add_argument("--pick_model", choices=["last_model", "val_acc", "val_loss"])

args = parser.parse_args()

if args.use_cuda:
    assert torch.cuda.is_available(), "CUDA is unavailable. Please use CPU instead."

start_time = time.clock()

TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = data_util.load_dataset(args)

preloading_time = time.clock()
print("Preloading time: {}".format(preloading_time - start_time))


def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)


def train_model(model, train_iter, epoch, attr):
    total_epoch_loss = 0
    total_epoch_acc = 0
    if args.use_cuda:
        model.cuda()
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    steps = 0
    model.train()
    for idx, batch in enumerate(train_iter):
        text = batch.问题现象[0]
        target = torch.Tensor(getattr(batch, attr)).long()
        if args.use_cuda:
            text = text.cuda()
            target = target.cuda()
        if args.model != "CNN":
            if text.size()[
                0] != args.batch_size:  # One of the batch returned by BucketIterator has length different than batch size.
                continue
        optim.zero_grad()
        prediction = model(text)
        # print(prediction)
        # print(target)
        loss = loss_fn(prediction, target)
        # print(loss)
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects / len(batch)
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1

        if steps % 100 == 0:
            print(
                f'Epoch: {epoch + 1}, Idx: {idx + 1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .2f}%')

        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()

    return total_epoch_loss / len(train_iter), total_epoch_acc / len(train_iter)


def eval_model(model, val_iter, attr):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            text = batch.问题现象[0]
            target = torch.Tensor(getattr(batch, attr)).long()
            if args.model != "CNN":
                if text.size()[0] != args.batch_size:
                    continue
            if args.use_cuda:
                text = text.cuda()
                target = target.cuda()
            prediction = model(text)
            loss = loss_fn(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects / len(batch)
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss / len(val_iter), total_epoch_acc / len(val_iter)


def choose_model(args, word_embeddings, output_size):
    if args.model == "LSTM":
        from models.LSTM import LSTMClassifier
        model = LSTMClassifier(args.batch_size, output_size, args.hidden_size, vocab_size, args.embed_size,
                               word_embeddings, args)
    elif args.model == "LSTM_Attn":
        from models.LSTM_Attn import AttentionModel
        model = AttentionModel(args.batch_size, output_size, args.hidden_size, vocab_size, args.embed_size,
                               word_embeddings, args)
    elif args.model == "RCNN":
        from models.RCNN import RCNN
        model = RCNN(args.batch_size, output_size, args.hidden_size, vocab_size, args.embed_size,
                     word_embeddings, args)
    elif args.model == "RNN":
        from models.RNN import RNN
        model = RNN(args.batch_size, output_size, args.hidden_size, vocab_size, args.embed_size,
                    word_embeddings, args)
    elif args.model == "SelfAttention":
        from models.selfAttention import SelfAttention
        model = SelfAttention(args.batch_size, output_size, args.hidden_size, vocab_size, args.embed_size,
                              word_embeddings, args)
    elif args.model == "CNN":
        from models.CNN import CNN
        # task_size > 1 for multi-task classifier. Other models are not implemented for multi-task classification yet.
        model = CNN(args.batch_size, output_size, args.in_channels, args.out_channels, args.kernel_heights,
                    args.stride, args.padding, args.keep_probab, vocab_size, args.embed_size, word_embeddings,
                    task_size=args.task_size)
    if args.use_cuda:
        model = model.cuda()
    return model


best_model_list = []
attr_list = ['异常类型', '形成原因', '故障组件']

# for attr in attr_list:
attr = attr_list[0]
model_list = []
val_acc_list = []
val_loss_list = []
model_index = args.num_epochs
model = choose_model(args, word_embeddings, data_util.label_len(attr))
loss_fn = F.cross_entropy
print(data_util.label_len(attr))
for epoch in range(args.num_epochs):
    print("Start training on attribute {}.".format(attr))
    train_loss, train_acc = train_model(model, train_iter, epoch, attr)
    val_loss, val_acc = eval_model(model, valid_iter, attr)
    model_list.append(model)
    val_acc_list.append(val_acc)
    val_loss_list.append(val_loss)
    print(
        f'Epoch: {epoch + 1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')

if args.pick_model == "last_model":
    # Do nothing, so the last epoch's model will be selected.
    pass
elif args.pick_model == "val_acc":
    # Select model with highest val_acc as final model.
    model_index = val_acc_list.index(max(val_acc_list))
elif args.pick_model == "val_loss":
    # Select model with lowest val_loss as final model.
    model_index = val_loss_list.index(min(val_loss_list))

best_model = model_list[model_index]
best_model_list.append(best_model)
print("Select model in epoch {} with {} criteria as final model.".format(model_index + 1, args.pick_model))

train_time = time.clock()
print("Training time: {}".format(train_time - start_time))

test_loss, test_acc = eval_model(best_model, test_iter, attr)
print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')

test_example = "8月14日14:15-16:35，国网蒙东电力I6000所有系统监控状态异常。期间，系统健康运行时长及在线用户数指标缺失，URL探测异常。"

test_example = TEXT.preprocess(test_example)
test_example = [[TEXT.vocab.stoi[x] for x in test_example]]

test_example = np.asarray(test_example)
test_example = torch.LongTensor(test_example)
with torch.no_grad():
    test_tensor = test_example.clone().detach()
if args.use_cuda:
    test_tensor = test_tensor.cuda()
model.eval()
output = model(test_tensor, 1)
out = F.softmax(output, 1)

print(data_util.label_denumericalize(torch.argmax(out[0]), attr))

total_time = time.clock()
print("Finish testing. Total time: {}".format(total_time - start_time))
