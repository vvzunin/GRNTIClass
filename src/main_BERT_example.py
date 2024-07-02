import os
from prepare_datasets_BERT import get_grnti1_2_BERT_dataframes, prepair_model_datasets
from train_test_BERT import train_save_bert, test_save_results
from TrainSettings import TrainSettings

n = 0 
base_name = "results//"

for number_of_delteted_values in [11]:
    for minimal_number_of_elements_RGNTI2 in [50]:
        for minimal_number_of_words in [10]:
            path_info_before_save = base_name + f"data_info_from_bert {n}\\"
            if not os.path.exists(path_info_before_save):
                os.makedirs(path_info_before_save)
            df, df_test, n_classes = get_grnti1_2_BERT_dataframes("..\\datasets\\base\\ru\\raw", 
                                                      number_of_delteted_values=number_of_delteted_values, 
                                           minimal_number_of_elements_RGNTI2=minimal_number_of_elements_RGNTI2,
                                           minimal_number_of_words=minimal_number_of_words,
                                           dir_name=path_info_before_save)
            for max_number_tokens in [512]:
                for pre_trained_model_name in ['DeepPavlov/rubert-base-cased']:
                    for r in [16]:
                        for lora_alpha in [32]:
                            for lora_dropout in [0.05]:


                                train_dataset, validation_dataset, test_dataset,\
                                loss_fuction, model = prepair_model_datasets(df, df_test, n_classes,
                                                        max_number_tokens=max_number_tokens, 
                                                        pre_trained_model_name=pre_trained_model_name,
                                                            r=r,
                                                            lora_alpha=lora_alpha,
                                                            lora_dropout=lora_dropout)
                                for epoch in [1]:
                                    for batch_size in [8]:
                                        for weight_decay in [1e-6]:
                                            for warmup_steps in [10]:
                                                for fp16 in [True]:
                                                    for optim in ["adamw_bnb_8bit"]:
                                                            dir_name = base_name + f"model bert lora {n}\\"
                                                            sett = TrainSettings()

                                                            sett.settings["number_of_delteted_values"] = number_of_delteted_values
                                                            sett.settings["minimal_number_of_elements_RGNTI2"] = minimal_number_of_elements_RGNTI2
                                                            sett.settings["minimal_number_of_words"] = minimal_number_of_words
                                                            sett.settings["max_number_tokens"] = max_number_tokens
                                                            sett.settings["pre_trained_model_name"] = pre_trained_model_name
                                                            sett.settings["r"] = r
                                                            sett.settings["lora_alpha"] = lora_alpha
                                                            sett.settings["lora_dropout"] = lora_dropout
                                                            sett.settings["epoch"] = epoch
                                                            sett.settings["batch_size"] = batch_size
                                                            sett.settings["weight_decay"] = weight_decay
                                                            sett.settings["warmup_steps"] = warmup_steps
                                                            sett.settings["fp16"] = fp16
                                                            sett.settings["optim"] = optim
                                                            sett.save(path = dir_name)
                                                            n +=1 
                                                            merged_model = train_save_bert(train_dataset, 
                                                                                        validation_dataset, 
                                                                                            loss_fuction, 
                                                                                            model,
                                                                                            n_classes,
                                                                                            dir_name=dir_name, 
                                                                                            epoch=epoch,
                                                                                            batch_size=batch_size,
                                                                                            weight_decay=weight_decay,
                                                                                            warmup_steps=warmup_steps,
                                                                                            fp16=fp16,
                                                                                            optim=optim)
                                                            test_save_results(merged_model, 
                                                                              test_dataset, 
                                                                              n_classes, 
                                                                              dir_name)

