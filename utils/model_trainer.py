import torch
import numpy as np
from collections import Counter


class ModelTrainer:

    @staticmethod
    def train_one_epoch(data_loader, model, loss_f, optimizer, 
                        scheduler, epoch_idx, device, log_interval, max_epoch,
                        logger):
        model.train()  ## 

        num_cls = model.fc.out_features
        conf_mat = np.zeros((num_cls, num_cls))
        loss_sigma = []
        loss_mean = 0
        acc_avg = 0
        path_error = []
        for i, data in enumerate(data_loader):

            inputs, labels, path_imgs = data
            # inputs, labels = data   # batch
            inputs, labels = inputs.to(device), labels.to(device)

            # forward & backward
            outputs = model(inputs)
            loss = loss_f(outputs.cpu(), labels.cpu())
            optimizer.zero_grad()  ###
            loss.backward()
            optimizer.step()

            # 统计loss
            loss_sigma.append(loss.item())
            loss_mean = np.mean(loss_sigma)

            # 统计混淆矩阵
            _, predicted = torch.max(outputs.data, 1)
            for j in range(len(labels)):  # per sample
                cate_i = labels[j].cpu().numpy()
                pred_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pred_i] += 1.
                if cate_i != pred_i:
                    path_error.append((cate_i, pred_i, path_imgs[j]))
            acc_avg = conf_mat.trace() / conf_mat.sum()

            # 每10个iteration 打印一次训练信息
            if i % log_interval == log_interval - 1:
                logger.info("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".
                            format(epoch_idx + 1, max_epoch, i + 1, len(data_loader), loss_mean, acc_avg))
        # print("epoch:{} sampler: {}".format(epoch_idx, Counter(label_list)))
        
        scheduler.step()

        return loss_mean, acc_avg, conf_mat, path_error

    @staticmethod
    def valid_one_epoch(data_loader, model, loss_f, device):
        model.eval()

        num_cls = model.fc.out_features
        conf_mat = np.zeros((num_cls, num_cls))
        loss_sigma = []
        path_error = []

        for i, data in enumerate(data_loader):
            inputs, labels, path_imgs = data
            # inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
            loss = loss_f(outputs.cpu(), labels.cpu())

            # 统计混淆矩阵
            _, predicted = torch.max(outputs.data, 1)
            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pred_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pred_i] += 1.
                if cate_i != pred_i:
                    path_error.append((cate_i, pred_i, path_imgs[j]))
            # 统计loss
            loss_sigma.append(loss.item())

        acc_avg = conf_mat.trace() / conf_mat.sum()

        return np.mean(loss_sigma), acc_avg, conf_mat, path_error

