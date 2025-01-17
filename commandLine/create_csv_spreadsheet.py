import pandas as pd
import numpy as np

LEVEL_ALL = 0
LEVEL_1 = 1
LEVEL_2 = 2
LEVEL_3 = 3

def generate_prediction_results(predictions, text_ids, language, threshold,
                                normalize, correctness, output_path, grnti_level):
    rows = []
    for i, pred in enumerate(predictions):
        code_probs = [
            (code, f"{prob:.2f}")
            for code, prob in sorted(pred.items(), key=lambda x: x[1], reverse=True)
        ][:3]
        while len(code_probs) < 3:
            code_probs.append(("EMPTY", "EMPTY"))

        if grnti_level == LEVEL_ALL:
            probability = "/".join([prob for _, prob in code_probs])
        elif grnti_level == LEVEL_1:
            probability = code_probs[0][1]
        elif grnti_level == LEVEL_2:
            probability = code_probs[1][1]
        elif grnti_level == LEVEL_3:
            probability = code_probs[2][1]
        else:
            probability = "INVALID_LEVEL"

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
