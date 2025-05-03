import torch
import pickle
import json
from prepare_datasets_BERT import prepair_compute_metrics
from transformers import TrainingArguments, Trainer
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchmetrics.classification import MultilabelF1Score
from ignite.metrics import ClassificationReport
from ignite.engine.engine import Engine

from ignite.metrics.precision import Precision
from ignite.metrics.recall import Recall
from ignite.metrics.metrics_lambda import MetricsLambda

def train_save_bert(dataset_train, dataset_valid, 
                    loss_fuction_for_multiclass_classification, 
                    model,
                    n_classes,
                    dir_name="модель_bert_lora", 
                    epoch=5,
                    batch_size=8,
                    weight_decay=1e-6,
                    warmup_steps=10,
                    fp16=True,
                    optim="adamw_bnb_8bit"):
    compute_metrics = prepair_compute_metrics(n_classes=n_classes)

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss = loss_fuction_for_multiclass_classification(logits, labels)

            return (loss, outputs) if return_outputs else loss
    
    training_args = TrainingArguments(
        output_dir=dir_name,          
        num_train_epochs=epoch,
        per_device_train_batch_size = batch_size ,
        per_device_eval_batch_size = batch_size,
        warmup_steps=warmup_steps,
        logging_dir=dir_name,
        weight_decay=weight_decay,
        evaluation_strategy='epoch',
        save_strategy= "epoch",
        logging_strategy="steps",
        logging_steps = 100,
        load_best_model_at_end=True,
        save_total_limit=2,
        report_to='tensorboard',
        overwrite_output_dir = False,
        save_safetensors = False,
        fp16=fp16,
        optim = optim)

    # Создание trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_valid,
        compute_metrics=compute_metrics
    )
    torch.cuda.empty_cache()
    #Обучение
    trainer.train()
    # merge весов их сохранение 
    merged_model_multilabel = model.merge_and_unload()

    merged_model_multilabel.save_pretrained(dir_name + "\\model_merged")
    return merged_model_multilabel

def make_predictions(model, dataset_test, device):
    model.eval()
    y_pred_list = []
    model.to(device)
    for batch in tqdm(dataset_test):
            inputs = torch.tensor([batch['input_ids']]).to(device)
            mask = torch.tensor([batch['attention_mask']]).to(device)
            y_test = torch.tensor([batch['labels']]).to(device)

            with torch.no_grad():
                output = model(input_ids = inputs, attention_mask = mask, labels=y_test)
            
            # Move logits and labels to CPU
            logits = output.logits.detach().cpu()#.numpy()


            logits_flatten = torch.sigmoid(logits).numpy()#.flatten()

            y_pred_list.extend(logits_flatten)

    return y_pred_list


def test_save_results(model, daaset_test, n_classes, dir_name):
    # Получаем пердсказания
    pred_res = make_predictions(model, daaset_test, "cuda")

    treshold_list = [0.4 + 0.025 * x for x in range(24)]
    f1_score_macro_list = []
    for treshold in tqdm(treshold_list):
        multilabel_f1_score_macro = MultilabelF1Score(num_labels=n_classes, average='macro', threshold=treshold)

        f1_score_macro_list.append(multilabel_f1_score_macro(torch.tensor(pred_res), 
                                                            torch.tensor(daaset_test['labels'])))
    f1_score_micro_list = []
    for treshold in tqdm(treshold_list):
        multilabel_f1_score_micro = MultilabelF1Score(num_labels=n_classes, average='micro', threshold=treshold)

        f1_score_micro_list.append(multilabel_f1_score_micro(torch.tensor(pred_res), 
                                                            torch.tensor(daaset_test['labels'])))
        
    f1_score_weighted_list = []
    for treshold in tqdm(treshold_list):
        multilabel_f1_score_weighted = MultilabelF1Score(num_labels=n_classes, average='weighted', threshold=treshold)

        f1_score_weighted_list.append(multilabel_f1_score_weighted(torch.tensor(pred_res), 
                                                            torch.tensor(daaset_test['labels'])))
        

    plt.plot(treshold_list, f1_score_macro_list, label = "macro")
    plt.plot(treshold_list, f1_score_micro_list, label = "micro")
    plt.plot(treshold_list, f1_score_weighted_list, label = "weighted")
    plt.xticks(treshold_list, rotation=70)
    plt.title("Зависимость f1_score от threshold")
    plt.xlabel("threshold")
    plt.ylabel("f1_score")
    plt.legend()
    plt.grid()
    plt.savefig(dir_name + "Зависимость f1_score от threshold.png",
                    bbox_inches='tight')
    
    def eval_step(engine, batch):
        return batch

    default_evaluator = Engine(eval_step)
    metric = ClassificationReport(output_dict=True, is_multilabel=True)
    weighted_metric_precision = Precision(average='weighted', is_multilabel=True)
    weighted_metric_recall= Recall(average='weighted', is_multilabel=True)

    preds = (torch.tensor(pred_res) > 0.5).int()
    input = torch.tensor(daaset_test['labels']).int()

    precision = Precision(average=False, is_multilabel=True)
    recall = Recall(average=False, is_multilabel=True)
    F1 = precision * recall * 2 / (precision + recall + 1e-20)
    freq = torch.tensor([sum(input[:, i]) for i in range(input.shape[1])])#.tolist()
    # weights_per_class = input.shape[0] / (torch.tensor([el if el > 0 else 1 for el in freq])* input.shape[1])
    F1_wieghted = MetricsLambda(lambda t: torch.sum(t * freq).item() / input.shape[0], F1) # 

    metric.attach(default_evaluator, "cr")
    weighted_metric_precision.attach(default_evaluator, "weighted precision")
    weighted_metric_recall.attach(default_evaluator, "weighted recall")
    F1_wieghted.attach(default_evaluator, "weighted F1")



    state = default_evaluator.run([[preds, input]])
    result = state.metrics['cr']
    result['weighted precision'] = state.metrics['weighted precision']

    result['weighted recall'] = state.metrics['weighted recall']
    result['weighted F1'] = state.metrics['weighted F1']

    with open(dir_name + "test_results.json", "w") as outfile:
        json.dump(result, outfile)
