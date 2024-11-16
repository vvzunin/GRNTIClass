from prediction import prepair_model, prepair_data_level1, prepair_data_level2,\
prepair_dataset, make_predictions, save_rubrics_names
import torch

if __name__ == "__main__":
    df_test = prepair_data_level1("test_ru.csv")

    model1 = prepair_model(n_classes=31, lora_model_path = "src\\expriment_save_model")

    dataset_test = prepair_dataset(df_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    predictions_level1 = make_predictions(model1, dataset_test, device=device, threshold=0.5)
    save_rubrics_names(predictions_level1, path_to_csv = "result1.csv")


    del model1 
    torch.cuda.empty_cache()
    print("Part for second level")

    model2 = prepair_model(n_classes=246, lora_model_path = "src\\expriment_save_model2")



    df_test2 = prepair_data_level2(df_test, predictions_level1)

    dataset_test2 = prepair_dataset(df_test2)


    predictions_level2 = make_predictions(model2, dataset_test2, device=device, threshold=0.5)


    save_rubrics_names(predictions_level2, path_to_csv = "result2.csv")
