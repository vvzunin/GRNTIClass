def run_dialog(prog, loadJSON):
  print("Будет доступно в следующих версиях!")
  pass
  # from prediction import (
  #   prepair_model,
  #   prepair_data_level1,
  #   prepair_data_level2,
  #   prepair_dataset,
  #   make_predictions,
  #   save_rubrics,
  #   toRubrics,
  # )
  # from tqdm import tqdm
  # import torch
  # import os
  # from messages import printMessage

  # torch.cuda.empty_cache()
  # printMessage("libs")
    
  # config = loadJSON(prog["configPath"])
  # print(config["models"][config["modelType"]]["1"])
  # model1 = (
  #   None
  #   if config["models"][config["modelType"]]["1"] == ""
  #   else prepair_model(
  #     n_classes=36, lora_model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), config["models"][config["modelType"]]["1"])
  #   )
  # )  
  # model2 = (
  #   None
  #   if config["models"][config["modelType"]]["2"] == ""
  #   else prepair_model(
  #     n_classes=246,
  #     lora_model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), config["models"][config["modelType"]]["2"])
  #   )
  # )
  # model3 = (
  #   None
  #   if config["models"][config["modelType"]]["3"] == ""
  #   else prepair_model(
  #     n_classes=0,
  #     lora_model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), config["models"][config["modelType"]]["3"]),
  #   )
  # )
  # printMessage("modelsLoaded")
  
  # params = {
  #   "input_file": input_file,
  #   "output_file": output_file,
  #   "input_encode": input_encode,
  #   "output_encode": output_encode,
  #   "level": identifier,
  #   "packet": packet,
  #   "format": input_format,
  #   "language": language,
  #   "threshold": threshold,
  #   "normalisation": normalization,
  #   "device": device,
  #   "workers": workers,
  # }

  # printMessage("startPredict")
  # df_test = prepair_data_level1(
  #   input_file, format=input_format, encoding=input_encode
  # )
  # device = torch.device(params["device"] if torch.cuda.is_available() else "cpu")
  # printMessage("device", "ru", (device,))

  # if normalization != "not":
  #   printMessage("badFlag", "ru", ("-n", normalization))
  #   quit()

  # for i in tqdm(range(0, df_test.shape[0], packet)):
  #   dataset_loader = prepair_dataset(
  #     df_test.iloc[i : i + packet], workers=params["workers"]
  #   )
  #   predictions_level1 = make_predictions(model1, dataset_loader, device=device)
  #   if identifier == "RGNTI1":
  #     predictions_level1 = toRubrics(
  #       os.path.join(os.path.dirname(os.path.abspath(__file__)), config["models"][config["modelType"]]["1"]),
  #       predictions_level1,
  #       threshold,
  #     )
  #     save_rubrics(
  #       df_test.iloc[i : i + packet],
  #       predictions_level1,
  #       params,
  #       prog,
  #       i == 0,
  #       output_encode,
  #     )
  #   else:
  #     df_test2 = prepair_data_level2(
  #       os.path.dirname(os.path.abspath(__file__)),
  #       os.path.join(os.path.dirname(os.path.abspath(__file__)), config["models"][config["modelType"]]["1"]),
  #       df_test.iloc[i : i + packet],
  #       predictions_level1,
  #       threshold,
  #     )
  #     dataset_loader2 = prepair_dataset(df_test2, workers=params["workers"])
  #     predictions_level2 = make_predictions(
  #       model2, dataset_loader2, device=device
  #     )
  #     if identifier == "RGNTI2":
  #       predictions_level2 = toRubrics(
  #         os.path.join(os.path.dirname(os.path.abspath(__file__)), config["models"][config["modelType"]]["2"]),
  #         predictions_level2,
  #         threshold,
  #       )
  #       save_rubrics(
  #         df_test2,
  #         predictions_level2,
  #         params,
  #         prog,
  #         i == 0,
  #         output_encode,
  #       )
  #     else:
  #       printMessage("notComplete")
  #       predictions_level2 = toRubrics(
  #         os.path.join(os.path.dirname(os.path.abspath(__file__)), config["models"][config["modelType"]]["2"]),
  #         predictions_level2,
  #         threshold,
  #       )
  #       save_rubrics(
  #         df_test2,
  #         predictions_level2,
  #         params,
  #         prog,
  #         i == 0,
  #         output_encode,
  #       )

  # del model1
  # del model2
  # del model3
  # printMessage("finish")