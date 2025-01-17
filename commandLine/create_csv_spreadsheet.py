import pandas as pd
import numpy as np

LEVEL_ALL = 0
LEVEL_1 = 1
LEVEL_2 = 2
LEVEL_3 = 3

def normalize_grnti_level(grnti_list, n_elements=15):
    subset = grnti_list[:n_elements]
    total = sum(subset)
    if total == 0:
        return [0] * len(subset)
    normalized = [value / total for value in subset]
    return normalized + grnti_list[n_elements:]

def check_grnti_level(grnti_level, grnti_data):
    if grnti_level == LEVEL_ALL:
        probability = "/".join([f"{value:.2f}" for value in grnti_data[:3]])
    elif grnti_level == LEVEL_1:
        probability = f"{grnti_data[0]:.2f}"
    elif grnti_level == LEVEL_2:
        probability = f"{grnti_data[1]:.2f}"
    elif grnti_level == LEVEL_3:
        probability = f"{grnti_data[2]:.2f}"
    else:
        probability = "INVALID_LEVEL"
    return probability

def generate_prediction_results(predictions, text_ids, language, threshold,
                                normalize, correctness, output_path, grnti_level,
                                grnti_data, normalize_grnti=False):
    rows = []

    if normalize_grnti:
        grnti_data = [
            normalize_grnti_level(level) for level in grnti_data
        ]

    for i, pred in enumerate(predictions):
        code_probs = [
            (code, f"{prob:.2f}")
            for code, prob in sorted(pred.items(), key=lambda x: x[1], reverse=True)
        ][:3]
        while len(code_probs) < 3:
            code_probs.append(("EMPTY", "EMPTY"))

        if normalize_grnti:
            probability = check_grnti_level(grnti_level, grnti_data[i])
        else:
            probability = check_grnti_level(grnti_level, [float(prob) for _, prob in code_probs])

        result_status = "REJECT" if any(prob == "EMPTY" for _, prob in code_probs) else ""

        row = [
            text_ids[i],
            probability,
            language,
            threshold,
            normalize,
            correctness[i],
            result_status,
        ]
        rows.append(row)

    column_names = [
        "ID of text",
        "Probability",
        "Language",
        "Threshold",
        "Normalize",
        "Correct",
        "Result Status",
    ]

    df = pd.DataFrame(rows, columns=column_names)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
