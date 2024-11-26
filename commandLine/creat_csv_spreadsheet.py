import pandas as pd
import numpy as np

def generate_prediction_results(predictions, text_ids, language, threshold,
                                normalize, correctness, output_path):

    rows = []
    for i, pred in enumerate(predictions):

        code_probs = [
            (code, f"{prob:.2f}")
            for code, prob in sorted(pred.items(), key=lambda x: x[1], reverse=True)
        ][:3]
        while len(code_probs) < 3:
            code_probs.append(("EMPTY", "EMPTY"))


        flattened = [item for sublist in code_probs for item in sublist]


        result_status = "REJECT" if any(prob == "EMPTY" for _, prob in code_probs) else ""

        row = [
            text_ids[i],
            *flattened,
            language,
            threshold,
            normalize,
            correctness[i],
            result_status,
        ]
        rows.append(row)

    # Create the DataFrame
    column_names = [
        "ID of text",
        "Code1", "Probability1",
        "Code2", "Probability2",
        "Code3", "Probability3",
        "Language",
        "Threshold",
        "Normalize",
        "Correct",
        "Result Status",
    ]
    df = pd.DataFrame(rows, columns=column_names)

    # Save to CSV
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
