from torchmetrics.classification import (
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
    MultilabelStatScores
)
import torch
import xlsxwriter
import json
import copy

import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm2
tqdm2.pandas()


def _create_excel_formats(workbook):
    """Создает форматы для Excel файла"""
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
    merge_format_empty = workbook.add_format({
        'border': 0,
        'align': 'center',
        'valign': 'vcenter',
    })
    return merge_format, merge_format_not_bold, merge_format_empty


def _print_prediction_statistics(preds):
    """Выводит статистику по предсказаниям"""
    preds_best_treshold = torch.sum(preds >= 1e-5, axis=1, dtype=torch.int)
    preds_best_treshold_no_zeros_sum = torch.sum(preds_best_treshold >= 1)
    reject_number = preds.shape[0] - preds_best_treshold_no_zeros_sum

    print("Cтатистика количества пркдсказываемых классов"
          " при заданном threshold:")
    print("Среднее число предсказываемых классов для одной статьи,"
          " для которой получено предсказание",
          torch.sum(preds_best_treshold)/preds_best_treshold.shape[0])
    print("Минимальное число предсказываемых классов для одной статьи,"
          " для которой получено предсказание",
          torch.min(preds_best_treshold))
    print("Максимальное число предсказываемых классов для одной статьи,"
          " для которой получено предсказание",
          torch.max(preds_best_treshold))
    print("Количество отказов от классификации (Reject)", reject_number)
    print("Доля отказов от классификации (Reject)",
          1 - preds_best_treshold_no_zeros_sum / preds_best_treshold.shape[0])

    return reject_number


def _calculate_metrics_for_value(preds_new, test_dataset_labels,
                                 n_classes, threshold):
    """Рассчитывает все метрики для заданного значения порога"""
    # F1-мера макро
    multilabel_f1_score_macro = MultilabelF1Score(
        num_labels=n_classes, average='macro', threshold=threshold)
    multilabel_precision_macro = MultilabelPrecision(
        num_labels=n_classes, average='macro', threshold=threshold)
    multilabel_recall_macro = MultilabelRecall(
        num_labels=n_classes, average='macro', threshold=threshold)

    f1_macro = multilabel_f1_score_macro(preds_new, test_dataset_labels)
    precision_macro = multilabel_precision_macro(preds_new,
                                                 test_dataset_labels)
    recall_macro = multilabel_recall_macro(preds_new, test_dataset_labels)

    # F1-мера микро
    multilabel_f1_score_micro = MultilabelF1Score(
        num_labels=n_classes, average='micro', threshold=threshold)
    multilabel_precision_micro = MultilabelPrecision(
        num_labels=n_classes, average='micro', threshold=threshold)
    multilabel_recall_micro = MultilabelRecall(
        num_labels=n_classes, average='micro', threshold=threshold)

    f1_micro = multilabel_f1_score_micro(preds_new, test_dataset_labels)
    precision_micro = multilabel_precision_micro(preds_new,
                                                 test_dataset_labels)
    recall_micro = multilabel_recall_micro(preds_new, test_dataset_labels)

    # F1-мера взвешенная
    multilabel_f1_score_weighted = MultilabelF1Score(
        num_labels=n_classes, average='weighted', threshold=threshold)
    multilabel_precision_weighted = MultilabelPrecision(
        num_labels=n_classes, average='weighted', threshold=threshold)
    multilabel_recall_weighted = MultilabelRecall(
        num_labels=n_classes, average='weighted', threshold=threshold)

    f1_weighted = multilabel_f1_score_weighted(preds_new, test_dataset_labels)
    precision_weighted = multilabel_precision_weighted(preds_new,
                                                       test_dataset_labels)
    recall_weighted = multilabel_recall_weighted(preds_new,
                                                 test_dataset_labels)

    return {
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'precision_micro': precision_micro,
        'precision_weighted': precision_weighted,
        'recall_macro': recall_macro,
        'recall_micro': recall_micro,
        'recall_weighted': recall_weighted
    }


def _apply_top_k_prediction(preds, k):
    """Применяет top-k предсказание"""
    pred_for_top_k = torch.zeros(preds.shape, dtype=float)
    top_indeces = torch.topk(preds, k).indices
    preds_range = torch.arange(pred_for_top_k.size(0)).unsqueeze(1)
    pred_for_top_k[preds_range, top_indeces] = 1.
    return pred_for_top_k


def _update_best_metrics(current_metrics, best_metrics, best_some_f1,
                         metric_value, kind_of_metric):
    """Обновляет лучшие метрики если текущие лучше"""
    sum_f1 = (current_metrics['f1_macro'] +
              current_metrics['f1_micro'] +
              current_metrics['f1_weighted'])

    if sum_f1 > best_some_f1:
        best_some_f1 = sum_f1
        best_metrics.clear()
        best_metrics[f"best_{kind_of_metric}"] = metric_value

        for key, value in current_metrics.items():
            best_metrics[key] = value.detach().item()

    return best_some_f1


def _save_plots(list_for_metric_kind, metrics_lists, kind_of_metric, dir_name):
    """Сохраняет графики зависимости метрик"""
    # График F1-меры
    plt.figure()
    plt.plot(list_for_metric_kind, metrics_lists['f1_macro'], label="macro")
    plt.plot(list_for_metric_kind, metrics_lists['f1_micro'], label="micro")
    plt.plot(list_for_metric_kind, metrics_lists['f1_weighted'],
             label="weighted")
    plt.xticks(list_for_metric_kind, rotation=70)
    plt.title(f"Зависимость f1_score от {kind_of_metric}")
    plt.xlabel(kind_of_metric)
    plt.ylabel("f1_score")
    plt.legend()
    plt.grid()
    plt.savefig(dir_name + f"Зависимость f1_score от {kind_of_metric}.png",
                bbox_inches='tight')
    plt.close()

    # График Precision
    plt.figure()
    plt.plot(list_for_metric_kind, metrics_lists['precision_macro'],
             label="macro")
    plt.plot(list_for_metric_kind, metrics_lists['precision_micro'],
             label="micro")
    plt.plot(list_for_metric_kind, metrics_lists['precision_weighted'],
             label="weighted")
    plt.xticks(list_for_metric_kind, rotation=70)
    plt.title(f"Зависимость precision от {kind_of_metric}")
    plt.xlabel(kind_of_metric)
    plt.ylabel("precision")
    plt.legend()
    plt.grid()
    plt.savefig(dir_name + f"Зависимость precision от {kind_of_metric}.png",
                bbox_inches='tight')
    plt.close()

    # График Recall
    plt.figure()
    print(f"recall_micro_list {kind_of_metric}:",
          metrics_lists['recall_micro'])
    print(f"recall_macro_list {kind_of_metric}:",
          metrics_lists['recall_macro'])
    print(f"recall_wighted_list {kind_of_metric}:",
          metrics_lists['recall_weighted'])
    plt.plot(list_for_metric_kind, metrics_lists['recall_macro'],
             label="macro")
    plt.plot(list_for_metric_kind, metrics_lists['recall_micro'],
             label="micro")
    plt.plot(list_for_metric_kind, metrics_lists['recall_weighted'],
             label="weighted")
    plt.xticks(list_for_metric_kind, rotation=70)
    plt.title(f"Зависимость recall от {kind_of_metric}")
    plt.xlabel(kind_of_metric)
    plt.ylabel("recall")
    plt.legend()
    plt.grid()
    plt.savefig(dir_name + f"Зависимость recall от {kind_of_metric}.png",
                bbox_inches='tight')
    plt.close()


def _save_empty_plot_for_threshold(preds, list_for_metric_kind, kind_of_metric,
                                   dir_name):
    """Сохраняет график доли пустых результатов для порогового метода"""
    plt.figure()
    plt.plot(list_for_metric_kind,
             [torch.sum(
                 torch.sum(preds >= treshold, axis=1) < 0.1) / preds.shape[0]
              for treshold in tqdm(list_for_metric_kind)])
    plt.xticks(list_for_metric_kind, rotation=70)
    plt.title("Зависимость доли элементов с пустым"
              "результатом классификации от threshold")
    plt.xlabel(kind_of_metric)
    plt.ylabel("Доля Empty")
    plt.legend()
    plt.grid()
    plt.savefig(dir_name + "Зависимость доли элементов с пустым"
                " результатом классификации от threshold.png",
                bbox_inches='tight')
    plt.close()


def _calculate_per_class_metrics(preds_new, test_dataset_labels, n_classes,
                                 threshold):
    """Рассчитывает метрики для каждого класса отдельно"""
    multilabel_f1_score_none = MultilabelF1Score(num_labels=n_classes,
                                                 average='none',
                                                 threshold=threshold)
    multilabel_f1_score_none_res = multilabel_f1_score_none(
        preds_new,
        test_dataset_labels)

    multilabel_precision_none = MultilabelPrecision(num_labels=n_classes,
                                                    average='none',
                                                    threshold=threshold)
    multilabel_precision_none_res = multilabel_precision_none(
        preds_new,
        test_dataset_labels)

    multilabel_recall_none = MultilabelRecall(
        num_labels=n_classes, average='none', threshold=threshold)
    multilabel_recall_none_res = multilabel_recall_none(
        preds_new, test_dataset_labels)

    mulitlabel_stat_scores_none = MultilabelStatScores(
        num_labels=n_classes, average='none', threshold=threshold)
    mulitlabel_stat_scores_none_res = mulitlabel_stat_scores_none(
        preds_new, test_dataset_labels)[:, :-1]

    return {
        'f1': multilabel_f1_score_none_res,
        'precision': multilabel_precision_none_res,
        'recall': multilabel_recall_none_res,
        'stats': mulitlabel_stat_scores_none_res
    }


def _create_metrics_dataframe(per_class_metrics, n_classes,
                              grnti_mapping_dict_true_numbers_reverse):
    """Создает DataFrame с метриками для каждого класса"""
    return pd.DataFrame({
        "№": [grnti_mapping_dict_true_numbers_reverse[key] for
              key in range(n_classes)],
        "F1": torch.round(per_class_metrics['f1'], decimals=3),
        "Precision": torch.round(per_class_metrics['precision'], decimals=3),
        "Recall": torch.round(per_class_metrics['recall'], decimals=3),
        "TP": per_class_metrics['stats'][:, 0],
        "FP": per_class_metrics['stats'][:, 1],
        "TN": per_class_metrics['stats'][:, 2],
        "FN": per_class_metrics['stats'][:, 3]
    }).sort_values(by=['№'], ascending=True)


def _write_excel_results(worksheet, first_index, best_metrics, df_metrics,
                         kind_of_metric, kind_index, formats):
    """Записывает результаты в Excel"""
    merge_format, merge_format_not_bold, merge_format_empty = formats

    # Заголовок для усредненных метрик
    best_metric_value = best_metrics.pop('best_' + kind_of_metric)
    worksheet.merge_range(first_index + 0, 0, first_index + 0,
                          len(best_metrics)-2,
                          "Усредненные метрики для лучшего значения"
                          f" {kind_of_metric} = {round(best_metric_value, 3)}",
                          merge_format)

    # Записываем значения метрик
    for col, (key, val) in enumerate(best_metrics.items()):
        worksheet.write_string(first_index + 1, col, key,
                               merge_format_not_bold)
        if not kind_index:
            worksheet.set_column(first_index + 1, col, 16)
        worksheet.write(first_index + 2, col, round(val, 3),
                        merge_format_not_bold)
        worksheet.write_blank(first_index + 3, col, None, merge_format_empty)

    # Заголовок для метрик по классам
    worksheet.merge_range(first_index + 4, 0, first_index + 4, 7,
                          "Метрики для всех классов для"
                          f" лучшего значения {kind_of_metric}",
                          merge_format)

    # Заголовки столбцов
    for col, name in enumerate(df_metrics.columns):
        worksheet.write_string(first_index + 5, col, name,
                               merge_format_not_bold)

    # Данные метрик для каждого класса
    for index, row in df_metrics.iterrows():
        worksheet.write_string(first_index + 6 + index, 0, row[0],
                               merge_format_not_bold)
        for column in range(1, df_metrics.shape[1]):
            worksheet.write(first_index + 6 + index, column,
                            round(row[column], 3), merge_format_not_bold)

    # Пустая строка для разделения
    for column in range(df_metrics.shape[1]):
        worksheet.write_blank(first_index + 6 + df_metrics.shape[0],
                              column, None, merge_format_empty)


def _process_metric_type(preds, test_dataset_labels, n_classes, kind_of_metric,
                         list_for_metric_kind, dir_name, grnti_path, level,
                         worksheet, first_index, kind_index, formats):
    """Обрабатывает один тип метрики (threshold или top_k)"""
    # Инициализация списков для метрик
    metrics_lists = {
        'f1_macro': [], 'f1_micro': [], 'f1_weighted': [],
        'precision_macro': [], 'precision_micro': [], 'precision_weighted': [],
        'recall_macro': [], 'recall_micro': [], 'recall_weighted': []
    }

    best_metrics = {}
    best_some_f1 = 0
    preds_new = copy.deepcopy(preds)

    # Проходим по всем значениям метрик
    for metric_value in tqdm(list_for_metric_kind):
        if kind_of_metric == "top_k":
            threshold = 0.
            preds_new = _apply_top_k_prediction(preds, metric_value)
        else:
            threshold = metric_value

        # Рассчитываем метрики
        current_metrics = _calculate_metrics_for_value(preds_new,
                                                       test_dataset_labels,
                                                       n_classes, threshold)

        # Добавляем в списки
        for key, value in current_metrics.items():
            metrics_lists[key].append(value)

        # Обновляем лучшие метрики
        best_some_f1 = _update_best_metrics(current_metrics,
                                            best_metrics, best_some_f1,
                                            metric_value, kind_of_metric)

    # Применяем лучшее значение метрики
    best_metric_value = best_metrics[f"best_{kind_of_metric}"]
    if kind_of_metric == "top_k":
        threshold = 0.
        preds_new = _apply_top_k_prediction(preds, best_metric_value)
    else:
        threshold = best_metric_value
        empty_number = preds.shape[0] - torch.sum(
            (torch.sum(preds, axis=1) - threshold) > 1e-5,
            dtype=torch.int).item()
        print("Доля статей c пустым ответом классификатора (Empty):",
              1 - empty_number / preds.shape[0])

    # Сохраняем графики
    _save_plots(list_for_metric_kind, metrics_lists, kind_of_metric, dir_name)

    if kind_of_metric == "threshold":
        _save_empty_plot_for_threshold(preds, list_for_metric_kind,
                                       kind_of_metric, dir_name)

    # Рассчитываем метрики для каждого класса
    per_class_metrics = _calculate_per_class_metrics(preds_new,
                                                     test_dataset_labels,
                                                     n_classes, threshold)

    # Загружаем маппинг ГРНТИ
    with open(grnti_path + f'my_grnti{level}_int.json', "r") as code_file:
        grnti_mapping_dict_true_numbers = json.load(code_file)

    grnti_mapping_dict_true_numbers_reverse = {
        y: x for x, y in grnti_mapping_dict_true_numbers.items()}

    # Создаем DataFrame
    df_metrics = _create_metrics_dataframe(
        per_class_metrics, n_classes, grnti_mapping_dict_true_numbers_reverse)

    # Добавляем суммы в best_metrics
    sum_of_tp_fp_tn_fn = [df_metrics[el].sum() for el
                          in ['TP', 'FP', 'TN', 'FN']]
    for key, val in zip(["TP sum", "FP sum", "TN sum", "FN sum"],
                        sum_of_tp_fp_tn_fn):
        best_metrics[key] = val

    # Записываем в Excel
    df_metrics["№"] = df_metrics["№"].astype(str)
    actual_first_index = first_index + kind_index * (7 + df_metrics.shape[0])
    _write_excel_results(worksheet, actual_first_index, best_metrics,
                         df_metrics, kind_of_metric, kind_index, formats)

    return empty_number if kind_of_metric == "threshold" else 0


def test_predictons(preds, test_dataset_labels, dir_name,
                    n_classes, level, grnti_path=""):
    """
    Функция для анализа качества предсказаний мультиклассовой модели
    и сохранения результатов в Excel и графики
    """
    # Создаем Excel-файл для сохранения результатов анализа
    workbook = xlsxwriter.Workbook(dir_name + 'Статистика тестирования.xlsx')
    formats = _create_excel_formats(workbook)
    worksheet = workbook.add_worksheet(f"BERT_RU_{level}_0")

    # Преобразуем предсказания в тензор
    preds = torch.tensor(preds)
    test_el_number = preds.shape[0]

    # Выводим статистику по предсказаниям
    reject_number = _print_prediction_statistics(preds)

    # Определяем списки порогов и топ-k для оценки
    threshold_list = [0.1 + 0.05 * x for x in range(0, 18, 1)]
    top_k_list = list(range(1, 4))

    # Преобразуем метки в тензор
    test_dataset_labels = torch.tensor(test_dataset_labels, dtype=torch.float)

    # Обрабатываем оба типа метрик
    empty_number = 0
    for kind_index, (kind_of_metric, list_for_metric_kind) in enumerate(
          zip(["threshold", "top_k"], [threshold_list, top_k_list])):

        result = _process_metric_type(preds, test_dataset_labels,
                                      n_classes, kind_of_metric,
                                      list_for_metric_kind, dir_name,
                                      grnti_path, level,
                                      worksheet, 4, kind_index, formats)
        if kind_of_metric == "threshold":
            empty_number = result

    # Добавляем общую информацию о количестве REJECT, EMPTY и документов
    worksheet.merge_range(0, 0, 0, 2,
                          "Количество ответов REJECT, EMPTY и документов",
                          formats[0])

    for index, (number, number_name) in enumerate(
        zip([reject_number, empty_number, test_el_number],
            ["кол-во ответов REJECT", "кол-во ответов EMPTY",
             "кол-во документов"])):
        worksheet.write_string(1, index, number_name, formats[2])
        worksheet.write(2, index, number, formats[1])

    # Закрываем файл Excel
    workbook.close()
