import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm2
tqdm2.pandas()


def load_and_preprocess_data(file_path):
    """
    Загружает и выполняет первичную обработку данных.

    Parameters:
    -----------
    file_path : str
        Путь к файлам данных.

    Returns:
    --------
    tuple
        (df, df_test) - обработанные тренировочные и тестовые данные
    """
    # Загрузка данных
    df = pd.read_csv(file_path + "/train_ru.csv", sep='\t', encoding='cp1251')
    df = df.loc[df['RGNTI'].apply(lambda x: re.findall(r"\d+", x) != [])]

    df_test = pd.read_csv(file_path + "/test_ru.csv",
                          sep='\t', encoding='cp1251', on_bad_lines='skip')
    df_test = df_test.loc[df_test['RGNTI'].apply(
        lambda x: re.findall(r"\d+", x) != [])]

    return df, df_test


def extract_grnti_levels(df, df_test):
    """
    Извлекает значения ГРНТИ для всех трех уровней.

    Parameters:
    -----------
    df : pd.DataFrame
        Тренировочные данные
    df_test : pd.DataFrame
        Тестовые данные

    Returns:
    --------
    tuple
        (df, df_test) - данные с добавленными столбцами
        target, target_2, target_3
    """
    # Извлекаем значения ГРНТИ 1-го уровня
    df['target'] = df['RGNTI'].apply(lambda x:
                                     list(set([re.findall(r"\d+", el)[0]
                                              for el in x.split('\\')])))
    df_test['target'] = df_test['RGNTI'].apply(
        lambda x: list(set([re.findall(r"\d+", el)[0] for
                            el in x.split('\\')])))

    # Извлекаем значения ГРНТИ 2-го уровня
    df['target_2'] = df['RGNTI'].apply(
        lambda x: list(set([re.findall(r"\d+\.\d+", el)[0] for
                            el in x.split('\\')
                            if re.findall(r"\d+\.\d+", el)])))

    df_test['target_2'] = df_test['RGNTI'].apply(
        lambda x: list(set([re.findall(r"\d+\.\d+", el)[0]
                            for el in x.split('\\')
                            if re.findall(r"\d+\.\d+", el)])))

    # Извлекаем значения ГРНТИ 3-го уровня
    df['target_3'] = df['RGNTI'].apply(
        lambda x: list(set([re.findall(r"\d+\.\d+\.\d+", el)[0] for
                            el in x.split('\\')
                            if re.findall(r"\d+\.\d+\.\d+", el)])))

    df_test['target_3'] = df_test['RGNTI'].apply(
        lambda x: list(set([re.findall(r"\d+\.\d+\.\d+", el)[0] for
                            el in x.split('\\')
                            if re.findall(r"\d+\.\d+\.\d+", el)])))

    return df, df_test


def prepare_text_data(df):
    """
    Подготавливает текстовые данные для обучения.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame с исходными данными

    Returns:
    --------
    pd.DataFrame
        DataFrame с подготовленными текстовыми данными
    """
    df['text'] = (
        df['title'].apply(lambda x: str(x) + ' [SEP] ') +
        df['ref_txt'].apply(lambda x: str(x)) +
        ' [SEP] ' + df['kw_list'].apply(lambda x: str(x)))

    return df


def apply_initial_filters(df, df_test, minimal_number_of_words):
    """
    Применяет начальные фильтры к данным.

    Parameters:
    -----------
    df : pd.DataFrame
        Тренировочные данные
    df_test : pd.DataFrame
        Тестовые данные
    minimal_number_of_words : int
        Минимальное количество слов в тексте

    Returns:
    --------
    tuple
        (df, df_test) - отфильтрованные данные
    """
    # Подготовка текстовых данных
    df = prepare_text_data(df)
    df_test = prepare_text_data(df_test)

    # Удаление пустых текстов
    df = df.dropna(subset=['text'], axis=0)
    df_test = df_test.dropna(subset=['text'], axis=0)

    # Фильтрация по минимальному количеству слов
    df = df[df['text'].apply(
        lambda x: len(str(x).split()) > minimal_number_of_words)]

    return df, df_test


def create_level_plots(data, level_num, dir_name, title_suffix,
                       figsize=(8, 10)):
    """
    Создает графики для анализа распределения ГРНТИ определенного уровня.

    Parameters:
    -----------
    data : pd.Series
        Данные для построения графика
    level_num : int
        Номер уровня ГРНТИ
    dir_name : str
        Директория для сохранения
    title_suffix : str
        Суффикс для названия графика
    figsize : tuple
        Размер фигуры
    """
    if not dir_name or data.empty:
        return

    df_plot = pd.DataFrame({
        f'RGNTI {level_num}': data.index,
        'Количество элементов': data.values
    })

    plt.figure(facecolor="#fff3e0", figsize=figsize, dpi=500)
    sns_plot = sns.barplot(
        y=f"RGNTI {level_num}", x="Количество элементов", data=df_plot)
    sns_plot.tick_params(labelsize=6 if level_num > 1 else 8)
    plt.xticks(fontsize=10)
    plt.tight_layout()
    plt.title(f"{title_suffix} для {level_num}-ого уровня ГРНТИ")
    plt.savefig(dir_name +
                f"{title_suffix} для {level_num}-ого уровня ГРНТИ.png",
                bbox_inches='tight')
    plt.close()


def get_valid_values_level1(df, target_col, min_elements):
    """
    Получает допустимые значения для 1-го уровня ГРНТИ.

    Parameters:
    -----------
    df : pd.DataFrame
        Тренировочные данные
    target_col : str
        Название столбца с целевыми значениями
    min_elements : int
        Минимальное количество элементов для сохранения

    Returns:
    --------
    tuple
        (valid_values, all_values) - допустимые значения
        и все значения с частотой
    """
    all_values = pd.Series(
        np.concatenate(df[target_col].values)).value_counts()

    if min_elements > 0:
        valid_values = set(
            all_values[all_values >= min_elements].index.tolist())
    else:
        valid_values = set(all_values.index.tolist())

    return valid_values, all_values


def get_valid_values_higher_levels(df, target_col, level_num,
                                   min_elements, parent_level_values):
    """
    Получает допустимые значения для 2-го и 3-го уровней ГРНТИ.

    Parameters:
    -----------
    df : pd.DataFrame
        Тренировочные данные
    target_col : str
        Название столбца с целевыми значениями
    level_num : int
        Номер уровня ГРНТИ (2 или 3)
    min_elements : int
        Минимальное количество элементов для сохранения
    parent_level_values : set
        Допустимые значения родительского уровня

    Returns:
    --------
    set
        Допустимые значения для данного уровня
    """
    valid_values = set()

    for parent_val in tqdm(parent_level_values):
        if level_num == 2:
            pattern = rf"{parent_val}\.\d+"
        else:  # level_num == 3
            pattern = rf"{parent_val}\.\d+"

        needed_values = df[target_col].apply(
            lambda x: [re.findall(pattern, el)[0]
                       for el in x if re.findall(pattern, el)])

        non_empty_lists = [el for el in needed_values.values.tolist() if el]
        if non_empty_lists:
            value_counts = pd.Series(
                np.concatenate(non_empty_lists)).value_counts()
            valid_values.update(
                value_counts[value_counts >= min_elements].index.tolist())

    return valid_values


def create_removed_elements_plot(df, target_col, level_num, valid_values,
                                 all_values, dir_name, min_elements):
    """
    Создает график удаляемых элементов.

    Parameters:
    -----------
    df : pd.DataFrame
        Данные для анализа
    target_col : str
        Название столбца с целевыми значениями
    level_num : int
        Номер уровня ГРНТИ
    valid_values : set
        Допустимые значения
    all_values : pd.Series, optional
        Все значения с частотой (для 1-го уровня)
    dir_name : str
        Директория для сохранения
    min_elements : int
        Минимальное количество элементов
    """
    if not dir_name:
        return

    if level_num == 1:
        # Для 1-го уровня используем готовые данные all_values
        if all_values is not None and len(all_values) > len(valid_values):
            removed_values = all_values[all_values < min_elements]
            if not removed_values.empty:
                create_level_plots(
                    removed_values, level_num, dir_name,
                    "Количество удаляемых текстов из датасета",
                    (8, 10))
    else:
        # Для 2-го и 3-го уровней
        if min_elements > 1:
            all_current_values = set(np.concatenate(df[target_col].values))
            removed_values_set = all_current_values - valid_values
            if removed_values_set:
                removed_counts = pd.Series(np.concatenate(
                    df[target_col].apply(
                        lambda x: list(set(x) & removed_values_set)).values
                )).value_counts()
                if not removed_counts.empty:
                    figsize = (8, 16) if level_num == 2 else (8, 18)
                    create_level_plots(
                        removed_counts, level_num, dir_name,
                        "Количество удаляемых текстов из датасета",
                        figsize)


def create_remaining_elements_plot(df_filtered, target_col,
                                   level_num, dir_name):
    """
    Создает график оставшихся элементов.

    Parameters:
    -----------
    df_filtered : pd.DataFrame
        Отфильтрованные данные
    target_col : str
        Название столбца с целевыми значениями
    level_num : int
        Номер уровня ГРНТИ
    dir_name : str
        Директория для сохранения
    """
    if not dir_name:
        return

    remaining_values = pd.Series(np.concatenate(
        df_filtered[target_col].values)).value_counts()
    figsize_map = {1: (8, 14), 2: (8, 16), 3: (8, 18)}
    create_level_plots(remaining_values, level_num, dir_name,
                       "Количество элементов, остающихся в датасете",
                       figsize_map[level_num])


def apply_data_filtering(df, df_test, target_col, valid_values):
    """
    Применяет фильтрацию к данным на основе допустимых значений.

    Parameters:
    -----------
    df : pd.DataFrame
        Тренировочные данные
    df_test : pd.DataFrame
        Тестовые данные
    target_col : str
        Название столбца с целевыми значениями
    valid_values : set
        Допустимые значения

    Returns:
    --------
    tuple
        (df_filtered, df_test_filtered) - отфильтрованные данные
    """
    df_filtered = df.copy()
    df_test_filtered = df_test.copy()

    # Фильтрация тренировочных данных
    df_filtered[target_col] = df[target_col].apply(
        lambda x: list(set(x) & valid_values))
    df_filtered = df_filtered[df_filtered[target_col].apply(lambda x: x != [])]

    # Фильтрация тестовых данных
    df_test_filtered[target_col] = df_test[target_col].apply(
        lambda x: list(set(x) & valid_values))
    df_test_filtered[target_col] = df_test_filtered[target_col].apply(
        lambda x: x if x != [] else ["no class"])

    return df_filtered, df_test_filtered


def filter_level_data(df, df_test, level_num, min_elements,
                      parent_level_values=None, dir_name=None):
    """
    Универсальная функция для фильтрации данных по любому уровню ГРНТИ.

    Parameters:
    -----------
    df : pd.DataFrame
        Тренировочные данные
    df_test : pd.DataFrame
        Тестовые данные
    level_num : int
        Номер уровня ГРНТИ (1, 2 или 3)
    min_elements : int
        Минимальное количество элементов для сохранения
    parent_level_values : set, optional
        Допустимые значения родительского уровня (для уровней 2 и 3)
    dir_name : str, optional
        Директория для сохранения графиков

    Returns:
    --------
    tuple
        (df_filtered, df_test_filtered, valid_values) -
        отфильтрованные данные и допустимые значения
    """
    target_col = f'target{"_" + str(level_num) if level_num > 1 else ""}'

    print(f"Удаление элементов {level_num}-го уровня, "
          f"количество которых меньше {min_elements}")

    # 1. Получение допустимых значений
    if level_num == 1:
        valid_values, all_values = get_valid_values_level1(
            df, target_col, min_elements)
    else:
        valid_values = get_valid_values_higher_levels(
            df, target_col, level_num, min_elements, parent_level_values)
        all_values = None

    # 2. Создание графика удаляемых элементов
    create_removed_elements_plot(df, target_col, level_num, valid_values,
                                 all_values, dir_name, min_elements)

    # 3. Применение фильтрации к данным
    df_filtered, df_test_filtered = apply_data_filtering(df, df_test,
                                                         target_col,
                                                         valid_values)

    # 4. Создание графика оставшихся элементов
    create_remaining_elements_plot(df_filtered, target_col, level_num,
                                   dir_name)

    return df_filtered, df_test_filtered, valid_values


def create_or_load_encoding(unique_values, level_num,
                            grnti_folder, change_codes):
    """
    Создает или загружает кодирование для ГРНТИ определенного уровня.

    Parameters:
    -----------
    unique_values : np.array
        Уникальные значения для кодирования
    level_num : int
        Номер уровня ГРНТИ
    grnti_folder : str
        Папка для файлов кодирования
    change_codes : bool
        Флаг для перезаписи файлов с кодами

    Returns:
    --------
    tuple
        (grnti_mapping_dict, n_classes) -
        словарь кодирования и количество классов
    """
    union_of_targets = set(unique_values)
    coding = range(len(union_of_targets))
    dict_Vinit_code_int = dict(zip(union_of_targets, coding))

    filename = f"my_grnti{level_num}_int.json"

    if change_codes:
        with open(grnti_folder + filename, "w") as outfile:
            json.dump(dict_Vinit_code_int, outfile)

    with open(grnti_folder + filename, "r") as code_file:
        grnti_mapping_dict = json.load(code_file)

    n_classes = len(grnti_mapping_dict)
    return grnti_mapping_dict, n_classes


def encode_targets(target_lists, mapping_dict, n_classes):
    """
    Вспомогательная функция для кодирования целевых меток.

    Parameters:
    -----------
    target_lists : list
        Списки целевых меток
    mapping_dict : dict
        Словарь кодирования
    n_classes : int
        Количество классов

    Returns:
    --------
    list
        Закодированные метки
    """
    result = []
    for list_el in target_lists:
        classes_zero = [0] * n_classes
        for index in list_el:
            if index in mapping_dict.keys():
                classes_zero[mapping_dict[index]] = 1
        result.append(classes_zero)
    return result


def get_grnti1_2_3_BERT_dataframes(file_path,
                                   number_of_deleted_values,
                                   minimal_number_of_elements_RGNTI2,
                                   minimal_number_of_elements_RGNTI3,
                                   minimal_number_of_words,
                                   dir_name=None,
                                   change_codes=False,
                                   grnti_folder=""):
    """
    Основная функция для обработки и фильтрации данных ГРНТИ трех уровней
    и создания графиков распределения элементов.

    Parameters:
    -----------
    file_path : str
        Путь к файлам данных.
    number_of_deleted_values : int
        Количество удаляемых значений ГРНТИ 1-го уровня.
    minimal_number_of_elements_RGNTI2 : int
        Минимальное количество элементов для ГРНТИ 2-го уровня.
    minimal_number_of_elements_RGNTI3 : int
        Минимальное количество элементов для ГРНТИ 3-го уровня.
    minimal_number_of_words : int
        Минимальное количество слов в тексте.
    dir_name : str, optional
        Директория для сохранения графиков. Если None, графики не сохраняются.
    change_codes : bool, default=False
        Флаг для перезаписи файлов с кодами ГРНТИ.
    grnti_folder : str, default=""
        Папка для хранения файлов соответствия кодов ГРНТИ.

    Returns:
    --------
    tuple
        (df_final, df_test_final, n_classes, n_classes2, n_classes3)
    """
    # 1. Загрузка и первичная обработка данных
    df, df_test = load_and_preprocess_data(file_path)

    # 2. Извлечение значений ГРНТИ всех уровней
    df, df_test = extract_grnti_levels(df, df_test)

    # 3. Применение начальных фильтров (перенесено в начало)
    df, df_test = apply_initial_filters(df, df_test, minimal_number_of_words)

    # 4. Фильтрация по уровням ГРНТИ с использованием универсальной функции

    # Фильтрация 1-го уровня
    df_level1, df_test_level1, valid_values_1 = filter_level_data(
        df, df_test, 1, number_of_deleted_values, dir_name=dir_name)

    # Фильтрация 2-го уровня
    df_level2, df_test_level2, valid_values_2 = filter_level_data(
        df_level1, df_test_level1, 2,
        minimal_number_of_elements_RGNTI2,
        valid_values_1, dir_name)

    # Фильтрация 3-го уровня
    df_final, df_test_final, valid_values_3 = filter_level_data(
        df_level2, df_test_level2, 3,
        minimal_number_of_elements_RGNTI3,
        valid_values_2, dir_name)

    # 5. Создание кодирования для всех уровней
    grnti_mapping_dict, n_classes = create_or_load_encoding(
        list(valid_values_1), 1, grnti_folder, change_codes)

    grnti_mapping_dict2, n_classes2 = create_or_load_encoding(
        list(valid_values_2), 2, grnti_folder, change_codes)

    grnti_mapping_dict3, n_classes3 = create_or_load_encoding(
        list(valid_values_3), 3, grnti_folder, change_codes)

    # 6. Кодирование классов для всех уровней
    df_final['target_coded'] = encode_targets(
        df_final['target'], grnti_mapping_dict, n_classes)
    df_final['target_coded2'] = encode_targets(
        df_final['target_2'], grnti_mapping_dict2, n_classes2)
    df_final['target_coded3'] = encode_targets(
        df_final['target_3'], grnti_mapping_dict3, n_classes3)

    # 7. Кодирование тестовых данных
    df_test_final['target_coded'] = encode_targets(
        df_test_final['target'], grnti_mapping_dict, n_classes)
    df_test_final['target_coded2'] = encode_targets(
        df_test_final['target_2'], grnti_mapping_dict2, n_classes2)
    df_test_final['target_coded3'] = encode_targets(
        df_test_final['target_3'], grnti_mapping_dict3, n_classes3)

    print("Доля оставшихся элементов в тренировочном датасете: ",
          df_final.shape[0] / df.shape[0])

    return df_final, df_test_final, n_classes, n_classes2, n_classes3
