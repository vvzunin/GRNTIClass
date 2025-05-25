import json
import os


class TrainSettings():
    def __init__(self):
        self.settings = {}

    def save(self, path='', filename="setting.json"):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path+filename, "w") as outfile:
            s = json.dumps(self.settings, indent=2, sort_keys=False)
            outfile.write(s)

    def load(self, path='', filename="setting.json"):
        with open(path+filename, "r") as infile:
            self.settings = json.loads(infile.read())
