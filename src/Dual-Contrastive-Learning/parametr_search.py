import torch
from tqdm import tqdm
from model import Transformer
# from config import get_config
from loss_func import CELoss, SupConLoss, DualLoss
from data_utils import load_data
from transformers import logging, AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

import numpy as np

import os
import sys
import time
import torch
import random
import logging
import argparse
from datetime import datetime


def get_config(data_dir="data",
               dataset="sst2",
               model_name="bert",
               method="dualcl",
               train_batch_size=16,
               num_epoch=100,
               lr=1e-5,
               decay=0.01,
               alpha = 0.5,
               temp=0.1,
               backend=False,
               timestamp = None,
               device='cuda' if torch.cuda.is_available() else 'cpu'
               ):
    if timestamp is None:
        timestamp = '{:.0f}{:03}'.format(time.time(), random.randint(0, 999)),

    # parser = argparse.ArgumentParser()
    num_classes = {'sst2': 2, 'subj': 2, 'trec': 6, 'pc': 2, 'cr': 2,
                   'rucola':5, 'rucola_2_labels':2}
    ''' Base '''
    args= dict()
    args["data_dir"] = data_dir
    args["dataset"] = dataset
    args["model_name"] = model_name
    args["method"] = method

    # parser.add_argument('--data_dir', type=str, default='data')
    # parser.add_argument('--dataset', type=str, default='sst2', choices=num_classes.keys())
    # parser.add_argument('--model_name', type=str, default='bert', choices=['bert',
    #                                                                        'roberta',
    #                                                                        'ruroberta'])
    # parser.add_argument('--method', type=str, default='dualcl', choices=['ce', 'scl', 'dualcl'])
    ''' Optimization '''
    args["train_batch_size"] = train_batch_size
    args["num_epoch"] = num_epoch
    args["lr"] = lr
    args["decay"] = decay
    args["alpha"] = alpha
    args["temp"] = temp

    # parser.add_argument('--train_batch_size', type=int, default=16)
    # parser.add_argument('--test_batch_size', type=int, default=64)
    # parser.add_argument('--num_epoch', type=int, default=100)
    # parser.add_argument('--lr', type=float, default=1e-5)
    # parser.add_argument('--decay', type=float, default=0.01)
    # parser.add_argument('--alpha', type=float, default=0.5)
    # parser.add_argument('--temp', type=float, default=0.1)
    ''' Environment '''
    args["backend"] = backend
    args["timestamp"] = timestamp
    args["device"] = device

    # parser.add_argument('--backend', default=False, action='store_true')
    # parser.add_argument('--timestamp', type=int, default='{:.0f}{:03}'.format(time.time(), random.randint(0, 999)))
    # parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # args = parser.parse_args()
    
    args.num_classes = num_classes[args.dataset]
    args.device = torch.device(args.device)
    args.log_name = '{}_{}_{}_{}.log'.format(args.dataset, args.model_name,
                                             args.method,
                                             datetime.now().strftime('%Y-%m-%d_%H-%M-%S')[2:])
    if not os.path.exists('logs'):
        os.mkdir('logs')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.addHandler(logging.FileHandler(os.path.join('logs', args.log_name)))
    return args, logger


class Instructor:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.logger.info('> creating model {}'.format(args.model_name))
        if args.model_name == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            base_model = AutoModel.from_pretrained('bert-base-uncased')
        elif args.model_name == 'roberta':
            self.tokenizer = AutoTokenizer.from_pretrained('roberta-base',
                                                           add_prefix_space=True)
            base_model = AutoModel.from_pretrained('roberta-base')
        elif args.model_name == 'ruroberta':
            self.tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/ruRoberta-large",
                                                           add_prefix_space=True)
            base_model = AutoModel.from_pretrained("sberbank-ai/ruRoberta-large")
        else:
            raise ValueError('unknown model')
        self.model = Transformer(base_model, args.num_classes, args.method)
        self.model.to(args.device)
        if args.device.type == 'cuda':
            self.logger.info('> cuda memory allocated: {}'.format(
                torch.cuda.memory_allocated(args.device.index)))
        self._print_args()

    def _print_args(self):
        self.logger.info('> training arguments:')
        for arg in vars(self.args):
            self.logger.info(f">>> {arg}: {getattr(self.args, arg)}")

    def _train(self, dataloader, criterion, optimizer):
        train_loss, n_correct, n_train = 0, 0, 0
        self.model.train()

        preds_list = []
        targets_list = []
        for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii=' >='):
            inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
            targets = targets.to(self.args.device)
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * targets.size(0)
            n_correct += (torch.argmax(outputs['predicts'], -1) == targets).sum().item()
            n_train += targets.size(0)

            preds_list.append(torch.argmax(outputs['predicts'], -1).cpu().numpy())
            targets_list.append(targets.cpu().numpy())

        targets_list = np.concatinate(targets_list)
        preds_list = np.concatinate(preds_list)

        # print(f"train {len(preds_list)} {preds_list[0].shape}")
        # print(f"train {len(targets_list)} {targets_list[0].shape}")

        f1 = f1_score(targets_list, preds_list, average='weighted')
        mcc = matthews_corrcoef(targets_list, preds_list)

        return train_loss / n_train, n_correct / n_train, f1, mcc

    def _test(self, dataloader, criterion):
        test_loss, n_correct, n_test = 0, 0, 0
        self.model.eval()

        preds_list = []
        targets_list = []
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii=' >='):
                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                targets = targets.to(self.args.device)
                outputs = self.model(inputs)
                
                loss = criterion(outputs, targets)
                test_loss += loss.item() * targets.size(0)
                n_correct += (torch.argmax(outputs['predicts'], -1) == targets).sum().item()
                n_test += targets.size(0)

                preds_list.append(torch.argmax(outputs['predicts'], -1).cpu().numpy())
                targets_list.append(targets.cpu().numpy())
                
        # accuracy = accuracy_score(targets_list, preds_list)
        # targets_list =
        targets_list = np.concatinate(targets_list)
        preds_list = np.concatinate(preds_list)

        # print(f"test {len(preds_list)} {preds_list[0].shape}")
        # print(f"test {len(targets_list)} {targets_list[0].shape}")

        f1 = f1_score(targets_list, preds_list, average='weighted')
        mcc = matthews_corrcoef(targets_list, preds_list)

        return test_loss / n_test, n_correct / n_test, f1, mcc

    def run(self):
        train_dataloader, test_dataloader = load_data(dataset=self.args.dataset,
                                                      data_dir=self.args.data_dir,
                                                      tokenizer=self.tokenizer,
                                                      train_batch_size=self.args.train_batch_size,
                                                      test_batch_size=self.args.test_batch_size,
                                                      model_name=self.args.model_name,
                                                      method=self.args.method,
                                                      workers=0)
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.args.method == 'ce':
            criterion = CELoss()
        elif self.args.method == 'scl':
            criterion = SupConLoss(self.args.alpha, self.args.temp)
        elif self.args.method == 'dualcl':
            criterion = DualLoss(self.args.alpha, self.args.temp)
        else:
            raise ValueError('unknown method')
        optimizer = torch.optim.AdamW(_params, lr=self.args.lr, weight_decay=self.args.decay)
        best_loss, best_mcc = 0, -1
        for epoch in range(self.args.num_epoch):
            train_loss, train_acc, train_f1, train_mcc  = self._train(train_dataloader, criterion, optimizer)
            test_loss, test_acc, test_f1, test_mcc = self._test(test_dataloader, criterion)


            if test_mcc > best_mcc or (test_mcc == best_mcc and test_loss < best_loss):
                best_mcc, best_loss = test_mcc, test_loss
                # self.model.save_pretrained('{}_{}_{}_pretrained_model'.format(args.dataset, 
                #                                                               args.model_name,
                #                                                               args.method))
                self.model.base_model.save_pretrained('{}_{}_{}_base_model'.format(args.dataset, 
                                                                              args.model_name,
                                                                              args.method))
                torch.save({
                    'linear_state_dict': self.model.linear.state_dict(),
                    'dropout_state_dict': self.model.dropout.state_dict(),
                }, '{}_{}_{}additional_layers.pth'.format(args.dataset,
                                                          args.model_name,
                                                          args.method))
                
                
            self.logger.info('{}/{} - {:.2f}%'.format(epoch+1, self.args.num_epoch, 100*(epoch+1)/self.args.num_epoch))
            self.logger.info('[train] loss: {:.4f}, acc: {:.2f}, f1: {:.2f}, mcc: {:.2f}'.format(train_loss, train_acc*100, train_f1*100,train_mcc*100))
            self.logger.info('[test] loss: {:.4f}, acc: {:.2f}, f1: {:.2f}, mcc: {:.2f}'.format(test_loss, test_acc*100, test_f1*100,test_mcc*100))
        self.logger.info('best loss: {:.4f}, best mcc: {:.2f}'.format(best_loss, best_mcc*100))
        self.logger.info('log saved: {}'.format(self.args.log_name))


if __name__ == '__main__':
    logging.set_verbosity_error()
    args, logger = get_config()
    ins = Instructor(args, logger)
    ins.run()