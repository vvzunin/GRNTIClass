import json
import re
import os


def openJsonFile(path):
    with open(path, 'r', encoding='utf-8') as f:
        a = json.load(f)
    return a


def openNormalFile(path):
    with open(path, encoding='utf-8') as f:
        a = f.read()
    return a


# Регулярные выражения для удаления формул
def remove_latex_formulas(text):
    # Удалить формулы внутри $$...$$
    text = re.sub(r'\$\$.*?\$\$', '', text, flags=re.DOTALL)
    # Удалить формулы внутри $...$
    text = re.sub(r'\$.*?\$', '', text, flags=re.DOTALL)
    # Удалить формулы внутри \[...\] или \(...\)
    text = re.sub(r'\\\[.*?\\\]', '', text, flags=re.DOTALL)
    # Удалить окружения формул (\begin{...}...\end{...})
    text = re.sub(r'\\begin\{.*?\}.*?\\end\{.*?\}', '', text, flags=re.DOTALL)
    return text


def output(text, pathToDir, outputName):
    if not os.path.isdir(pathToDir):
        print(f"The directory {pathToDir} doesn't exist.\nCreating...")
        os.mkdir(pathToDir)
        print(f"Directory {pathToDir} was created.")
    pathToFile = os.path.join(pathToDir, outputName)
    try:
        with open(pathToFile, 'w', encoding='utf-8') as f:
            f.write(text)
        print("File was successfully writen.")
    except:
        print("Save error. File wasn't written.")


def replaceWithDict(text, dict):
    for abbs in dict:
        text = text.replace(abbs, dict[abbs])
    return text

def preProcessing(pathToText, pathToDict, pathToDir="", outputName="output.txt"):
    text = openNormalFile(pathToText)

    dictOfAbbs = openJsonFile(pathToDict)

    preProcessingText = replaceWithDict(text, dictOfAbbs)

    clean_text = remove_latex_formulas(preProcessingText)
    output(text=clean_text, pathToDir=pathToDir, outputName=outputName)


preProcessing("text.txt", "abbreviations.json", pathToDir="_Output_", outputName="output.txt")
