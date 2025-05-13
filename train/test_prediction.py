# Стандартная библиотека Python
import json
import copy

# Сторонние библиотеки для обработки данных
import pandas as pd

# Визуализация
import matplotlib.pyplot as plt

# tqdm для прогресс-баров
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm2
# Активация прогресс-баров для pandas
tqdm2.pandas()

# Библиотеки для работы с Excel
import xlsxwriter

# Библиотеки для ML/DL
import torch
from torchmetrics.classification import (
    MultilabelF1Score,
    MultilabelPrecision, 
    MultilabelRecall, 
    MultilabelStatScores
)


def test_predictons(preds, test_dataset_labels, dir_name, 
                       n_classes, level,
                       grnti_path=""):
    """
    Функция для анализа качества предсказаний мультиклассовой модели
    и сохранения результатов в Excel и графики
    
    Аргументы:
    - preds: предсказания модели
    - test_dataset_labels: истинные метки тестового набора
    - dir_name: директория для сохранения результатов
    - n_classes: количество классов
    - level: уровень иерархии классификации
    - grnti_path: путь к файлу с маппингом кодов ГРНТИ
    """
    # Создаем Excel-файл для сохранения результатов анализа
    workbook  = xlsxwriter.Workbook(dir_name + 'Статистика тестирования.xlsx')

    # Определяем форматы для ячеек Excel
    merge_format = workbook.add_format({
        'bold': 1,
        'border': 1,
        'align': 'center',
        'valign': 'vcenter',
    })
    merge_format_not_bold = workbook.add_format({
        'border': 1,
        'align': 'center',
        'valign': 'vcenter',
    })
    merge_format_not_bold_small = workbook.add_format({
        'border': 1,
        'align': 'center',
        'valign': 'vcenter',
        'font_size': 9
    })

    merge_format_empty = workbook.add_format({
        'border': 0,
        'align': 'center',
        'valign': 'vcenter',
    })
    # Создаем лист в Excel
    worksheet = workbook.add_worksheet(f"BERT_RU_{level}_0")


    # Преобразуем предсказания в тензор
    preds = torch.tensor(preds)
    test_el_number = preds.shape[0]

    # Определяем количество классов, предсказанных для каждого документа (порог активации 1e-5)
    preds_best_treshold = torch.sum(preds >= 1e-5, axis = 1,
                                    dtype=torch.int)  # сумма для каждого класса
    
    # Считаем количество элементов с непустыми предсказаниями
    preds_best_treshold_no_zeros_sum =  torch.sum(preds_best_treshold >= 1) # кол-во элементов без пропусков
    # Количество отказов от классификации
    reject_number = preds.shape[0] - preds_best_treshold_no_zeros_sum

    # Выводим статистику по количеству предсказываемых классов
    print("Cтатистика количества пркдсказываемых классов при заданном threshold:")
    print("Среднее число предсказываемых классов для одной статьи, для которой получено предсказание",
            torch.sum(preds_best_treshold)/preds_best_treshold.shape[0])
    print("Минимальное число предсказываемых классов для одной статьи, для которой получено предсказание",
            torch.min(preds_best_treshold))
    print("Максимальное число предсказываемых классов для одной статьи,"
          " для которой получено предсказание",
            torch.max(preds_best_treshold))

    # Статистика отказов
    print("Количество отказов от классификации (Reject)", 
          reject_number)
    print("Доля отказов от классификации (Reject)", 
          1 - preds_best_treshold_no_zeros_sum / preds_best_treshold.shape[0])
    
    # Определяем списки порогов и топ-k для оценки
    threshold_list = [0.1 + 0.05 * x for x in range(0, 18, 1)]
    top_k_list = list(range(1, 4))
    preds_new = copy.deepcopy(preds)

    # Преобразуем метки в тензор
    test_dataset_labels = torch.tensor(test_dataset_labels, dtype=torch.float)

    # Проходим по двум типам метрик: пороговый (threshold) и топ-k (выбор топ-k классов)
    for kind_of_metric, list_for_metric_kind, kind_index in zip(["threshold", "top_k"],
                                                    [threshold_list, top_k_list],
                                                    [0, 1]):
        first_index = 4

        # Инициализируем списки для хранения значений метрик
        f1_score_macro_list = []
        f1_score_micro_list = []
        f1_score_weighted_list = []


        precision_macro_list = []
        precision_micro_list = []
        precision_weighted_list = []

        recall_macro_list = []
        recall_micro_list = []
        recall_weighted_list = []
        
        # Инициализируем переменные для сохранения лучших метрик
        best_metric = None
        best_metrics  = dict()
        best_some_f1 = 0

        # Проходим по всем значениям метрик (порогов или топ-k)
        for metric_values in tqdm(list_for_metric_kind):
            if kind_of_metric == "top_k":
                # Для топ-k: берем k классов с наивысшей вероятностью
                threshold = 0.
                pred_for_top_k = torch.zeros(preds.shape, dtype=float)  

                top_indeces = torch.topk(preds, metric_values).indices

                preds_range = torch.arange(pred_for_top_k.size(0)).unsqueeze(1)

                pred_for_top_k[preds_range, top_indeces] = 1.
                preds_new = pred_for_top_k
            else:
                # Для порогового подхода просто используем заданный порог
                threshold = metric_values
                
            # Рассчитываем F1-меру макро
            multilabel_f1_score_macro = MultilabelF1Score(num_labels=n_classes, average='macro', 
                                                        threshold=threshold)
            multilabel_precision_macro = MultilabelPrecision(num_labels=n_classes, average='macro', 
                                                        threshold=threshold)
            multilabel_recall_macro = MultilabelRecall(num_labels=n_classes, average='macro', 
                                                        threshold=threshold)
            
            f1_score_macro_list.append(multilabel_f1_score_macro(preds_new, 
                                                                test_dataset_labels))
            precision_macro_list.append(multilabel_precision_macro(preds_new,
                                                                test_dataset_labels))
            recall_macro_list.append(multilabel_recall_macro(preds_new,
                                                            test_dataset_labels))
            
            # Рассчитываем F1-меру микро
            multilabel_f1_score_micro = MultilabelF1Score(num_labels=n_classes, average='micro',
                                                        threshold=threshold)
            multilabel_precision_micro = MultilabelPrecision(num_labels=n_classes, average='micro', 
                                                        threshold=threshold)
            multilabel_recall_micro = MultilabelRecall(num_labels=n_classes, average='micro', 
                                                        threshold=threshold)
            
            f1_score_micro_list.append(multilabel_f1_score_micro(preds_new,
                                                                test_dataset_labels))
            
            precision_micro_list.append(multilabel_precision_micro(preds_new,
                                                                test_dataset_labels))
            recall_micro_list.append(multilabel_recall_micro(preds_new,
                                                            test_dataset_labels))

            # Рассчитываем F1-меру взвешенную
            multilabel_f1_score_weighted = MultilabelF1Score(num_labels=n_classes, average='weighted',
                                                            threshold=threshold)
            multilabel_precision_weighted  = MultilabelPrecision(num_labels=n_classes, average='weighted', 
                                                        threshold=threshold)
            multilabel_recall_weighted = MultilabelRecall(num_labels=n_classes, average='weighted', 
                                                        threshold=threshold)
            

            f1_score_weighted_list.append(multilabel_f1_score_weighted(preds_new,
                                                                    test_dataset_labels))
            precision_weighted_list.append(multilabel_precision_weighted(preds_new,
                                                                        test_dataset_labels))
            recall_weighted_list.append(multilabel_recall_weighted(preds_new,
                                                                test_dataset_labels))

            # Суммарная F1-мера для выбора лучшего значения метрики
            sum_f1 = f1_score_macro_list[-1] + f1_score_micro_list[-1] + f1_score_weighted_list[-1]

            # Если текущая сумма F1 лучше предыдущей, обновляем лучшие метрики
            if sum_f1 > best_some_f1:
                best_some_f1 = sum_f1
                best_metric = threshold if kind_of_metric == "threshold" else metric_values
                best_metrics[f"best_{kind_of_metric}"] = best_metric

                # Сохраняем лучшие значения метрик
                best_metrics["f1_macro"] = f1_score_macro_list[-1].detach().item()
                best_metrics["f1_micro"] = f1_score_micro_list[-1].detach().item()
                best_metrics["f1_weighted"] = f1_score_weighted_list[-1].detach().item()

                best_metrics["precision_macro"] = precision_macro_list[-1].detach().item()
                best_metrics["precision_micro"] = precision_micro_list[-1].detach().item()
                best_metrics["precision_weighted"] = precision_weighted_list[-1].detach().item()

                best_metrics["recall_macro"] = recall_macro_list[-1].detach().item()
                best_metrics["recall_micro"] = recall_micro_list[-1].detach().item()
                best_metrics["recall_weighted"] = recall_weighted_list[-1].detach().item()

        # Применяем лучшее значение метрики к предсказаниям
        if kind_of_metric == "top_k":
            # Для top-k: формируем предсказания с лучшим k
            threshold = 0.
            pred_for_top_k = torch.zeros(preds.shape, dtype=float)  

            top_indeces = torch.topk(preds, best_metric).indices

            preds_range = torch.arange(pred_for_top_k.size(0)).unsqueeze(1)

            pred_for_top_k[preds_range, top_indeces] = 1.
            preds_new = pred_for_top_k
        else:
            # Для порога: используем лучший порог
            threshold = best_metric
            
            # Считаем количество элементов с пустым результатом классификации
            empty_number = preds.shape[0] - torch.sum((torch.sum(preds, axis=1) - threshold) > 1e-5,
                                           dtype=torch.int).item()

            # Выводим долю статей с пустым ответом
            print("Доля статей c пустым ответом классификатора (Empty):",
                  1 - empty_number / preds.shape[0])

        # Рассчитываем метрики для каждого класса отдельно
        multilabel_f1_score_none = MultilabelF1Score(num_labels=n_classes, average='none',
                                                        threshold=threshold)
        multilabel_f1_score_none_res = multilabel_f1_score_none(preds_new,
                                                                test_dataset_labels)
        
        multilabel_precision_none = MultilabelPrecision(num_labels=n_classes, average='none',
                                                        threshold=threshold)
        multilabel_precision_none_res = multilabel_precision_none(preds_new,
                                                                test_dataset_labels)
        
        multilabel_recall_none = MultilabelRecall(num_labels=n_classes, average='none',
                                                        threshold=threshold)
        multilabel_recall_none_res = multilabel_recall_none(preds_new,
                                                            test_dataset_labels)
        
        # Получаем статистику TP, FP, TN, FN для каждого класса
        mulitlabel_stat_scores_none= MultilabelStatScores(num_labels=n_classes,
                                                          average='none',
                                                          threshold=threshold)
        
        mulitlabel_stat_scores_none_res = mulitlabel_stat_scores_none(preds_new,
                                                                    test_dataset_labels)[:, :-1]
        
        # Загружаем маппинг кодов ГРНТИ из JSON-файла
        with open(grnti_path + f'my_grnti{level}_int.json', "r") as code_file:
            grnti_mapping_dict_true_numbers = json.load(code_file) # Загружаем файл с кодами 

        # Создаем обратный словарь для маппинга индексов к кодам ГРНТИ
        grnti_mapping_dict_true_numbers_reverse = {y: x for x, y in 
                                                grnti_mapping_dict_true_numbers.items()}
        
        # Создаем DataFrame с метриками для каждого класса
        df_rubrics_f1_precision_recall_stats = pd.DataFrame({
            "№": [grnti_mapping_dict_true_numbers_reverse[key] for key in range(n_classes)], 
            "F1": torch.round(multilabel_f1_score_none_res, decimals=3),
            "Precision":torch.round(multilabel_precision_none_res, decimals=3),
            "Recall":torch.round(multilabel_recall_none_res, decimals=3),
            "TP":mulitlabel_stat_scores_none_res[:, 0],
            "FP":mulitlabel_stat_scores_none_res[:, 1],
            "TN":mulitlabel_stat_scores_none_res[:, 2],
            "FN":mulitlabel_stat_scores_none_res[:, 3]}).sort_values(by=['№'], ascending=True)

        # Строим график зависимости F1-меры от значения метрики
        plt.plot(list_for_metric_kind, f1_score_macro_list, label = "macro")
        plt.plot(list_for_metric_kind, f1_score_micro_list, label = "micro")
        plt.plot(list_for_metric_kind, f1_score_weighted_list, label = "weighted")
        plt.xticks(list_for_metric_kind, rotation=70)
        plt.title(f"Зависимость f1_score от {kind_of_metric}")
        plt.xlabel(kind_of_metric)
        plt.ylabel("f1_score")
        plt.legend()
        plt.grid()
        plt.savefig(dir_name + f"Зависимость f1_score от {kind_of_metric}.png",
                        bbox_inches='tight')
        plt.close()
        plt.figure()

        # Строим график зависимости точности (precision) от значения метрики
        plt.figure()
        plt.plot(list_for_metric_kind, precision_macro_list, label = "macro")
        plt.plot(list_for_metric_kind, precision_micro_list, label = "micro")
        plt.plot(list_for_metric_kind, precision_weighted_list, label = "weighted")
        plt.xticks(list_for_metric_kind, rotation=70)
        plt.title(f"Зависимость precision от {kind_of_metric}")
        plt.xlabel(kind_of_metric)
        plt.ylabel("precision")
        plt.legend()
        plt.grid()
        plt.savefig(dir_name + f"Зависимость precision от {kind_of_metric}.png",
                        bbox_inches='tight')
        plt.close()

        # Строим график зависимости полноты (recall) от значения метрики
        plt.figure()
        print(f"recall_micro_list {kind_of_metric}:", recall_micro_list)
        print(f"recall_macro_list {kind_of_metric}:", recall_macro_list)
        print(f"recall_wighted_list {kind_of_metric}:", recall_weighted_list)
        plt.plot(list_for_metric_kind, recall_macro_list, label = "macro")
        plt.plot(list_for_metric_kind, recall_micro_list, label = "micro")
        plt.plot(list_for_metric_kind, recall_weighted_list, label = "weighted")
        plt.xticks(list_for_metric_kind, rotation=70)
        plt.title(f"Зависимость recall от {kind_of_metric}")
        plt.xlabel(kind_of_metric)
        plt.ylabel("recall")
        plt.legend()
        plt.grid()
        plt.savefig(dir_name + f"Зависимость recall от {kind_of_metric}.png",
                        bbox_inches='tight')
        plt.close()

        # Для порогового подхода дополнительно строим график доли пустых результатов
        if kind_of_metric == "threshold":
            plt.figure()
            plt.plot(list_for_metric_kind,
                    [torch.sum(
                        torch.sum(preds >= treshold, axis = 1) < 0.1) / preds.shape[0] for 
                                    treshold in tqdm(list_for_metric_kind)
                                    ])
            plt.xticks(list_for_metric_kind, rotation=70)
            plt.title("Зависимость доли элементов с пустым результатом классификации "
                      f"от {kind_of_metric}")
            plt.xlabel(kind_of_metric)
            plt.ylabel("Доля Empty")
            plt.legend()
            plt.grid()
            plt.savefig(dir_name + "Зависимость доли элементов с пустым результатом"
                        f" классификации от {kind_of_metric}.png",
                            bbox_inches='tight')
            plt.close()

        # Считаем суммы TP, FP, TN, FN для всех классов
        sum_of_tp_fp_tn_fn = [df_rubrics_f1_precision_recall_stats[el].sum()
                              for el in ['TP', 'FP', 'TN', 'FN']]

        # Преобразуем номера рубрик в строки для Excel
        df_rubrics_f1_precision_recall_stats["№"] = df_rubrics_f1_precision_recall_stats["№"].\
            astype(str)
        first_index += kind_index * (7  + df_rubrics_f1_precision_recall_stats.shape[0])

        # Добавляем суммарные метрики в словарь лучших метрик
        for key, val in zip(["TP sum", "FP sum", "TN sum", "FN sum"], sum_of_tp_fp_tn_fn):
            best_metrics[key] = val

        # Записываем результаты в Excel: заголовок для усредненных метрик
        worksheet.merge_range(first_index + 0, 0, first_index + 0, len(best_metrics)-2,
                            "Усредненные метрики для лучшего значения "
                            f"{kind_of_metric} = {round(best_metrics['best_' + kind_of_metric], 3)}",
                            merge_format)

        # Удаляем использованный ключ из словаря
        best_metrics.pop('best_' + kind_of_metric)
        
        # Записываем значения метрик в Excel
        for col, key_val in enumerate(best_metrics.items()):
            worksheet.write_string(first_index + 1, col, key_val[0], merge_format_not_bold)  # Write the key as the header
            if not kind_index:
                worksheet.set_column(first_index + 1, col, 16) #len(key_val[0]) 15

            worksheet.write(first_index + 2, col, round(key_val[1], 3),merge_format_not_bold)  # Write the key as the header

            worksheet.write_blank(first_index + 3, col, None, merge_format_empty)

        # Записываем заголовок для метрик по классам
        worksheet.merge_range(first_index + 4, 0, first_index + 4, 7,
                            f"Метрики для всех классов для лучшего значения {kind_of_metric}",
                            merge_format)

        # Записываем заголовки столбцов
        for col, name in enumerate(df_rubrics_f1_precision_recall_stats.columns):
            worksheet.write_string(first_index + 5, col, name, merge_format_not_bold)

        # Записываем данные метрик для каждого класса
        for index, row in df_rubrics_f1_precision_recall_stats.iterrows():
            worksheet.write_string(first_index + 6 + index, 0, row[0], merge_format_not_bold)
            for column in range(1, df_rubrics_f1_precision_recall_stats.shape[1]):
                worksheet.write(first_index + 6 + index, column, round(row[column],3), merge_format_not_bold)

        # Добавляем пустую строку для разделения
        for column in range(df_rubrics_f1_precision_recall_stats.shape[1]):
            worksheet.write_blank(first_index + 6 + df_rubrics_f1_precision_recall_stats.shape[0],
                                  column, None, merge_format_empty)
                                  
    # Добавляем общую информацию о количестве REJECT, EMPTY и документов в начале листа
    worksheet.merge_range(0, 0, 0, 2,
                        "Количество ответов REJECT, EMPTY и документов",
                        merge_format)
    
    # Записываем значения количества REJECT, EMPTY и общего числа документов
    for index, number_number_name in enumerate(zip([reject_number, empty_number, test_el_number],
                                                    ["кол-во ответов REJECT", "кол-во ответов EMPTY", "кол-во документов"])):
        number, number_name = number_number_name
        worksheet.write_string(1, index, number_name, merge_format_not_bold_small)
        worksheet.write(2, index, number, merge_format_not_bold)

    # Закрываем файл Excel
    workbook.close()
