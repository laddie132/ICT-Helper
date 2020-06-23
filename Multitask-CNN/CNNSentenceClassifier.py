import torch
import torch.nn.functional as F
from models.CNN import CNN


def loss_fn(prediction, target):
    loss = []
    for index in range(0, len(prediction)):
        loss.append(F.cross_entropy(prediction[index], target.transpose(0, 1)[index]))
    return sum(loss)


class SentenceClassifier:
    def __init__(self, train_data_loader, dev_data_loader, test_data_loader,
                 batch_size, embedding_weight, reader,
                 use_cuda=False, k=3, in_channels=1, out_channels=128,
                 kernel_heights=None, stride=1, padding=0,
                 keep_probab=0.8, task_size=5, connection=None):
        if kernel_heights is None:
            kernel_heights = [3, 4, 5]
        if connection is None:
            connection = [-1, 0, -1, -1, 3]
        vocab_size = embedding_weight.size(0)
        embed_size = embedding_weight.size(1)

        self.train_data_loader = train_data_loader
        self.dev_data_loader = dev_data_loader
        self.test_data_loader = test_data_loader
        self.reader = reader
        self.label_itos = [label_vocab.labels for label_vocab in self.reader.label_vocab_list]
        self.output_size = [len(labels) for labels in self.label_itos]
        self.k = k
        self.task_size = task_size
        self.use_cuda = use_cuda
        self.model = CNN(batch_size, self.output_size, in_channels, out_channels, kernel_heights, stride,
                         padding, keep_probab, vocab_size, embed_size, embedding_weight, self.task_size, connection)

    def clip_gradient(self, clip_value):
        params = list(filter(lambda p: p.grad is not None, self.model.parameters()))
        for p in params:
            p.grad.data.clamp_(-clip_value, clip_value)

    def count_num_corrects(self, prediction, target):
        pred_index_list = []
        for index in range(0, len(prediction)):
            pred_index = prediction[index].topk(self.k)[1].squeeze()
            pred_index_list.append(pred_index.unsqueeze(1))
        pred_tensor = torch.cat(pred_index_list, dim=1)  # batch_size * label_size * k

        correct = torch.zeros(target.size(), dtype=torch.int8)  # batch_size * label_size
        for i in range(0, pred_tensor.size(0)):
            for j in range(0, pred_tensor.size(1)):
                # target size: batch_size * label_size
                if self.k == 1:
                    if target.data[i][j] == pred_tensor.data[i][j]:
                        correct[i][j] = 1
                else:
                    if target.data[i][j] in pred_tensor.data[i][j]:
                        correct[i][j] = 1
        num_corrects = correct.min(dim=1)[0].sum()

        label_acc = 100.0 * correct.sum(0).float() / self.train_data_loader.batch_size  # label_size
        for acc_index in range(0, label_acc.size(0)):
            print("label {} acc: {}".format(acc_index, label_acc[acc_index]))

        return num_corrects

    def logits2label(self, logits_list):
        pred_index_list = []
        for index in range(0, len(logits_list)):
            pred_index = logits_list[index].topk(self.k)[1].squeeze()
            if self.k == 1:
                pred_index_list.append(pred_index.unsqueeze(0))
                topk_output = torch.cat(pred_index_list, dim=0)  # label_size
            else:
                pred_index_list.append(pred_index.unsqueeze(1))
                topk_output = torch.cat(pred_index_list, dim=1).transpose(0, 1) # label_size * k
        label_output = {}
        keys = self.reader.KEYS
        for label_index in range(0, topk_output.size(0)):
            output_itos = []
            if self.k == 1:
                result = topk_output[label_index]
                output_itos.append(self.label_itos[label_index][result])
            else:
                for result_index in range(0, topk_output.size(1)):
                    # denumericalize label
                    result = topk_output[label_index][result_index]
                    output_itos.append(self.label_itos[label_index][result])
            label_output[keys[label_index]] = output_itos
        return label_output

    def train_epoch(self, epoch):
        total_epoch_loss = 0
        total_epoch_acc = 0
        if self.use_cuda:
            self.model.cuda()
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()))
        steps = 0
        self.model.train()
        for idx, batch in enumerate(self.train_data_loader):
            text = batch[0]
            target = batch[1]
            if self.use_cuda:
                text = text.cuda()
                target = target.cuda()
            optim.zero_grad()
            prediction = self.model(text)
            loss = loss_fn(prediction, target)
            num_corrects = self.count_num_corrects(prediction, target)
            acc = 100.0 * num_corrects / self.train_data_loader.batch_size
            loss.backward()
            self.clip_gradient(1e-1)
            optim.step()
            steps += 1

            if steps % 100 == 0:
                print(f'Epoch: {epoch + 1}, Idx: {idx + 1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .2f}%')

            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

        return total_epoch_loss / len(self.train_data_loader), total_epoch_acc / len(self.train_data_loader)

    def train(self, num_of_epoch):
        model_list = []
        val_acc_list = []

        for epoch in range(num_of_epoch):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.evaluate(self.dev_data_loader)
            model_list.append(self.model)
            val_acc_list.append(val_acc)
            print(f'Epoch: {epoch + 1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')

        model_index = val_acc_list.index(max(val_acc_list))
        self.model = model_list[model_index]

    def evaluate(self, data_loader):
        total_epoch_loss = 0
        total_epoch_acc = 0
        self.model.eval()
        if self.use_cuda:
            self.model = self.model.cuda()
        with torch.no_grad():
            for idx, batch in enumerate(data_loader):
                text = batch[0]
                target = batch[1]
                if self.use_cuda:
                    text = text.cuda()
                    target = target.cuda()
                prediction = self.model(text)
                loss = loss_fn(prediction, target)
                num_corrects = self.count_num_corrects(prediction, target)
                acc = 100.0 * num_corrects / data_loader.batch_size
                total_epoch_loss += loss.item()
                total_epoch_acc += acc.item()

        return total_epoch_loss / len(data_loader), total_epoch_acc / len(data_loader)

    def decode(self, text):
        text_tensor = self.reader.text_to_tensor(text).unsqueeze(0)
        if self.use_cuda:
            text_tensor = text_tensor.cuda()
        self.model.eval()
        logits_output = self.model(text_tensor, 1)
        final_output = self.logits2label(logits_output)
        return final_output

    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
