import json
import argparse
import datetime
from prediction import prepair_model, prepair_data_level1, prepair_data_level2, \
    prepair_dataset, make_predictions, save_rubrics, toRubrics
from tqdm import tqdm
import torch


with open("parameters.json", "r", encoding="utf-8") as f:
    parameters = json.load(f)

lang = "ru"


def printInfo(message, args=[]):
    print(message[lang].format(*args))

def parseArgs():
    parser = argparse.ArgumentParser(prog=parameters["prog"]["name"],
                                     description=parameters["descriptions"][lang])
    parser.add_argument('--version', action='version',
                        version=f"%(prog)s {parameters['prog']['version']}")

    for key, arg in parameters["arguments"].items():
        parser.add_argument(
            arg["name"],
            default=arg["default"],
            type=eval(arg["type"]),
            choices=arg["choices"],
            required=arg["required"],
            help=arg["help"][lang],
            metavar=arg["metavar"],
            dest=arg["dest"],
        )

    args = vars(parser.parse_args())
    return args


def dataSelection(preds, threshold):
    return preds[preds > threshold]


def main():
    start = datetime.datetime.now()
    printInfo(parameters["messages"]["start"], [start.strftime(parameters["datetime_format_output"])])

    user_args = parseArgs()

    torch.cuda.empty_cache()
    printInfo(parameters["messages"]["libs"], [datetime.datetime.now().strftime(parameters["datetime_format_output"])])

    model1 = None if not parameters["arguments"]["m"]["choices"] else prepair_model(n_classes=36, lora_model_path=user_args['m'])
    model2 = None
    model3 = None

    if user_args['level'] in ["RGNTI2", "RGNTI3"]:
        model2 = None if not parameters["arguments"]["m"]["choices"] else prepair_model(n_classes=246, lora_model_path=user_args['m'])
    if user_args['level'] == "RGNTI3":
        model3 = None if not parameters["arguments"]["m"]["choices"] else prepair_model(n_classes=0, lora_model_path=user_args['m'])

    if model1 is None or (model2 is None and user_args['level'] in ["RGNTI2", "RGNTI3"]) or (model3 is None and user_args['level'] == "RGNTI3"):
        printInfo(parameters["messages"]["model_error"], [datetime.datetime.now().strftime(parameters["datetime_format_output"])])
        exit()

    printInfo(parameters["messages"]["start_predict"], [datetime.datetime.now().strftime(parameters["datetime_format_output"])])
    df_test = prepair_data_level1(user_args['i'], format=user_args['f'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    printInfo(parameters["messages"]["device"], [datetime.datetime.now().strftime(parameters["datetime_format_output"]), device])

    if user_args['n'] != "not":
        printInfo(parameters["messages"]["bad_flag"], [datetime.datetime.now().strftime(parameters["datetime_format_output"]), '-n', user_args['n']])
        exit()

    for i in tqdm(range(df_test.shape[0])):
        dataset_loader = prepair_dataset(df_test.iloc[[i]])
        predictions_level1 = make_predictions(model1, dataset_loader, device=device)
        if user_args['level'] == "RGNTI1":
            predictions_level1 = toRubrics(predictions_level1, 1, user_args['t'])
            save_rubrics(df_test.iloc[[i]], predictions_level1, user_args, parameters["prog"], i == 0)
        else:
            df_test2 = prepair_data_level2(df_test.iloc[[i]], predictions_level1, user_args['t'])
            dataset_loader2 = prepair_dataset(df_test2)
            predictions_level2 = make_predictions(model2, dataset_loader2, device=device)
            if user_args['level'] == "RGNTI2":
                predictions_level2 = toRubrics(predictions_level2, 2, user_args['t'])
                save_rubrics(df_test2, predictions_level2, user_args, parameters["prog"], i == 0)
            else:
                printInfo(parameters["messages"]["not_complete"], [datetime.datetime.now().strftime(parameters["datetime_format_output"])])

    del model1
    del model2
    del model3

    printInfo(parameters["messages"]["finish"], [datetime.datetime.now().strftime(parameters["datetime_format_output"])])

if __name__ == "__main__":
    main()
