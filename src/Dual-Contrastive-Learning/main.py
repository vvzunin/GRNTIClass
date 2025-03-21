import torch
import time, random, sys, os

from tqdm import tqdm
from model import Transformer
# from config import get_config
from loss_func import CELoss, SupConLoss, DualLoss
from data_utils import load_data
from transformers import AutoTokenizer, AutoModel, logging
from logging import StreamHandler, FileHandler

from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from datetime import datetime
import numpy as np

def get_config(data_dir="data",
               dataset="grnit_level_1",
               model_name="rubert",
               method="dualcl",
               train_batch_size=16,
               test_batch_size = 64,
               num_epoch=50,#100
               lr=1e-5,
               decay=0.01,
               alpha = 0.5,
               temp=0.1,
               backend=False,
               timestamp = None,
               device='cuda' if torch.cuda.is_available() else 'cpu',
               class_weights = None):
    
    if timestamp is None:
        timestamp = '{:.0f}{:03}'.format(time.time(), random.randint(0, 999)),

    # parser = argparse.ArgumentParser()
    num_classes = {'sst2': 2, 'subj': 2, 'trec': 6, 'pc': 2, 'cr': 2,
                   'rucola':5, 'rucola_2_labels':2, 'grnit_level_1':32}
    ''' Base '''
    args= dict()
    args["data_dir"] = data_dir
    args["dataset"] = dataset
    args["model_name"] = model_name
    args["method"] = method

    ''' Optimization '''
    args["train_batch_size"] = train_batch_size
    args["test_batch_size"] = test_batch_size
    args["num_epoch"] = num_epoch
    args["lr"] = lr
    args["decay"] = decay
    args["alpha"] = alpha
    args["temp"] = temp


    ''' Environment '''
    args["backend"] = backend
    args["timestamp"] = timestamp
    args["device"] = device

    args["class_weights"] = class_weights

    args["num_classes"] = num_classes[args["dataset"]]
    args["device"] = torch.device(args["device"])
    args["log_name"] = '{}_{}_{}_{}.log'.format(args["dataset"], args["model_name"],
                                             args["method"],
                                             datetime.now().strftime('%Y-%m-%d_%H-%M-%S')[2:])
    if not os.path.exists('logs'):
        os.mkdir('logs')

    logger = logging.get_logger("transformers")#logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(StreamHandler(sys.stdout))
    logger.addHandler(FileHandler(os.path.join('logs', args["log_name"])))
    return args, logger


class Instructor:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.logger.info('> creating model {}'.format(args["model_name"]))

        if args["model_name"] == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            base_model = AutoModel.from_pretrained('bert-base-uncased')
        elif args["model_name"] == 'roberta':
            self.tokenizer = AutoTokenizer.from_pretrained('roberta-base',
                                                           add_prefix_space=True)
            base_model = AutoModel.from_pretrained('roberta-base')
        elif args["model_name"] == 'ruroberta':
            self.tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/ruRoberta-large",
                                                           add_prefix_space=True)
            base_model = AutoModel.from_pretrained("sberbank-ai/ruRoberta-large")
        elif args["model_name"] == "rubert": 
            self.tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
            base_model = AutoModel.from_pretrained('DeepPavlov/rubert-base-cased')
    
        else:
            raise ValueError('unknown model')
        self.model = Transformer(base_model, args["num_classes"], args["method"])
        self.model.to(args["device"])
        if args["device"].type == 'cuda':
            self.logger.info('> cuda memory allocated: {}'.format(
                torch.cuda.memory_allocated(args["device"].index)))
            
        self._print_args()

    def _print_args(self):
        self.logger.info('> training arguments:')
        # for arg in vars(self.args):
        for arg, value in self.args.items():

            self.logger.info(f">>> {arg}: {value}")#getattr(self.args, arg)

    def _train(self, dataloader, criterion, optimizer):
        train_loss, n_correct, n_train = 0, 0, 0
        self.model.train()

        preds_list = []
        targets_list = []
        for inputs, targets in tqdm(dataloader, disable=self.args["backend"], ascii=' >='):
            inputs = {k: v.to(self.args["device"]) for k, v in inputs.items()}
            targets = targets.to(self.args["device"])
            outputs = self.model(inputs)


            # print(f"{outputs=}")

            # Используем BCEWithLogitsLoss для многозадачной классификации
            loss = criterion(outputs, targets)#outputs['predicts']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * targets.size(0)
            
            # Преобразуем логиты в бинарные предсказания с порогом 0.5
            preds = (torch.sigmoid(outputs['predicts']) > 0.5).float()
            n_correct += (preds == targets).all(dim=1).sum().item()  # Считаем полностью правильные предсказания
            n_train += targets.size(0)

            preds_list.append(preds.cpu().numpy())
            targets_list.append(targets.cpu().numpy())

        targets_list = np.concatenate(targets_list)
        preds_list = np.concatenate(preds_list)

        # Рассчитываем метрики
        f1 = f1_score(targets_list, preds_list, average='samples')  # Используем 'samples' для многозадачной классификации
        accuracy = accuracy_score(targets_list, preds_list)
        mcc = matthews_corrcoef(targets_list, preds_list)

        return train_loss / n_train, n_correct / n_train, f1, accuracy, mcc

    def _test(self, dataloader, criterion):
        test_loss, n_correct, n_test = 0, 0, 0
        self.model.eval()

        preds_list = []
        targets_list = []
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, disable=self.args["backend"], ascii=' >='):
                inputs = {k: v.to(self.args["device"]) for k, v in inputs.items()}
                targets = targets.to(self.args["device"])
                outputs = self.model(inputs)
                
                # Используем BCEWithLogitsLoss для многозадачной классификации
                loss = criterion(outputs, targets)#outputs['predicts']
                test_loss += loss.item() * targets.size(0)
                
                # Преобразуем логиты в бинарные предсказания с порогом 0.5
                preds = (torch.sigmoid(outputs['predicts']) > 0.5).float()
                n_correct += (preds == targets).all(dim=1).sum().item()  # Считаем полностью правильные предсказания
                n_test += targets.size(0)

                preds_list.append(preds.cpu().numpy())
                targets_list.append(targets.cpu().numpy())
                
        targets_list = np.concatenate(targets_list)
        preds_list = np.concatenate(preds_list)

        # Рассчитываем метрики
        f1 = f1_score(targets_list, preds_list, average='samples')  # Используем 'samples' для многозадачной классификации
        accuracy = accuracy_score(targets_list, preds_list)
        mcc = matthews_corrcoef(targets_list, preds_list)

        return test_loss / n_test, n_correct / n_test, f1, accuracy, mcc, preds_list


    def run(self):
        train_dataloader, test_dataloader = load_data(dataset=self.args["dataset"],
                                                      data_dir=self.args["data_dir"],
                                                      tokenizer=self.tokenizer,#self.args["tokenizer"],
                                                      train_batch_size=self.args["train_batch_size"],
                                                      test_batch_size=self.args["test_batch_size"],
                                                      model_name=self.args["model_name"],
                                                      method=self.args["method"],
                                                      workers=0)
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.args["method"] == 'ce':
            criterion = CELoss()
        elif self.args["method"]== 'scl':
            criterion = SupConLoss(self.args["alpha"], self.args["temp"])
        elif self.args["method"]== 'dualcl':
            criterion = DualLoss(self.args["alpha"], self.args["temp"])
        else:
            raise ValueError('unknown method')
        optimizer = torch.optim.AdamW(_params, lr=self.args["lr"], weight_decay=self.args["decay"])
        best_loss, best_f1 = 0, 0

        for epoch in range(self.args["num_epoch"]):
            train_loss, train_acc, train_f1, train_mcc  = self._train(train_dataloader, criterion, optimizer)
            test_loss, test_acc, test_f1, test_mcc, preds_list = self._test(test_dataloader, criterion)


            if test_f1 > best_f1 or (test_f1 == best_f1 and test_loss < best_loss):
                best_f1, best_loss = test_f1, test_loss

                # self.model.base_model.save_pretrained(
                #     '{}_{}_{}_base_model'.format(
                #     self.args["dataset"], 
                #     self.args["model_name"],
                #     self.args["method"]))
                
                # torch.save({
                #     'linear_state_dict': self.model.linear.state_dict(),
                #     'dropout_state_dict': self.model.dropout.state_dict(),
                # }, '{}_{}_{}additional_layers.pth'.format(self.args["dataset"],
                #                                           self.args["model_name"],
                #                                           self.args["method"]))
                
                with open('{}_{}_{}_preds.npy'.format(
                    self.args["dataset"],
                    self.args["model_name"],
                    self.args["method"]), 'wb') as f:

                    np.save(f, preds_list)
                
            self.logger.info('{}/{} - {:.2f}%'.format(epoch+1, self.args["num_epoch"],
                                                      100*(epoch+1)/self.args["num_epoch"]))
            self.logger.info('[train] loss: {:.4f}, acc: {:.2f}, f1: {:.2f}, mcc: {:.2f}'.format(
                train_loss, train_acc*100, train_f1*100,train_mcc*100))
            self.logger.info('[test] loss: {:.4f}, acc: {:.2f}, f1: {:.2f}, mcc: {:.2f}'.format(
                test_loss, test_acc*100, test_f1*100,test_mcc*100))

        self.logger.info('best loss: {:.4f}, best f1: {:.2f}'.format(best_loss, best_f1*100))
    
        self.logger.info('log saved: {}'.format(self.args["log_name"]))


if __name__ == '__main__':
    logging.set_verbosity_error()
    args, logger = get_config()
    ins = Instructor(args, logger)
    ins.run()
