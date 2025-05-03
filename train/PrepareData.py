import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import sys
import swifter

sys.path.append('../')
from TrainSettings import TrainSettings
from tqdm import tqdm
settings = TrainSettings()
datasets_path = "../datasets/"

tqdm.pandas()


class PrepareData():
  def __init__(self, 
               settings,
               path='',
               train_filename='', 
               test_filename=''):
    if train_filename == '':
      self.train_filename = path + settings['DATASET_NAME'] + '/' + settings['LANG'] + '/' + settings['DATASET_VERSION'] + '/train_' + settings['LANG'] + '.csv'
    else:
      self.train_filename = train_filename
    
    if test_filename == '':
      self.test_filename = path + settings['DATASET_NAME'] + '/' + settings['LANG'] + '/' + settings['DATASET_VERSION'] + '/test_' + settings['LANG'] + '.csv'
    else:
      self.test_filename = test_filename

    self.settings = settings

  def readData(self, sep='\t'):
    self.df_train = pd.read_csv(self.train_filename, sep=sep, on_bad_lines='warn')
    self.df_test = pd.read_csv(self.test_filename, sep=sep, on_bad_lines='warn')

  def concatColumns(self):
    # Объеденяем поля с текстом
    self.df_train['text'] = (self.df_train['title'] if self.settings['DATASET_USE_TITLE'] else '') \
                    + self.df_train['body'] \
                    + (self.df_train['keywords'] if self.settings['DATASET_USE_KEYWORDS'] else '')
    self.df_train = self.df_train.drop("title", axis=1)
    self.df_train = self.df_train.drop("body", axis=1)
    self.df_train = self.df_train.drop("keywords", axis=1)    

    self.df_test['text'] = (self.df_test['title'] + ' ' if self.settings['DATASET_USE_TITLE'] else '') \
                    + self.df_test['body'] \
                    + (' ' + self.df_test['keywords'] if self.settings['DATASET_USE_KEYWORDS'] else '')
    self.df_test = self.df_test.drop("title", axis=1)
    self.df_test = self.df_test.drop("body", axis=1)
    self.df_test = self.df_test.drop("keywords", axis=1)
    
  def checkCorrect(self):
    # Удаляем невалидные данные
    self.df_train = self.df_train[self.df_train['correct'] == '###']
    self.df_test = self.df_test[self.df_test['correct'] == '###']

    # Удаляем лишние столбцы
    self.df_train = self.df_train.drop("correct", axis=1)
    self.df_test = self.df_test.drop("correct", axis=1)

  def chooseLevel(self):
    self.df_train['RGNTI'] = self.df_train[self.settings['LEVEL']].str.split('\\', n=1, expand=True)[0]
    self.df_test['RGNTI'] = self.df_test[self.settings['LEVEL']].str.split('\\', n=1, expand=True)[0]

    self.df_train = self.df_train.drop("RGNTI1", axis=1)
    self.df_train = self.df_train.drop("RGNTI2", axis=1)
    self.df_train = self.df_train.drop("RGNTI3", axis=1)
    self.df_train = self.df_train.dropna().drop_duplicates()

    self.df_test = self.df_test.drop("RGNTI1", axis=1)
    self.df_test = self.df_test.drop("RGNTI2", axis=1)
    self.df_test = self.df_test.drop("RGNTI3", axis=1)
    self.df_test = self.df_test.dropna().drop_duplicates()

  def removeTexts(self):
    # TODO: Сделать умное удаление, чтобы выборки распределены по уровням следующего уровня
    # Удаляем лишние тексты, чтобы избежать переобучения
    a = self.df_train.groupby('RGNTI').size().to_dict()
    t = []
    for i in a:
      if a[i] > self.settings['MAX_TEXTS'][self.settings['LEVEL']]:
        t.append(i)

    for i in t:
      ids = self.df_train.index[self.df_train['RGNTI'] == i].tolist()
      ids = ids[self.settings['MAX_TEXTS'][self.settings['LEVEL']]:]
      self.df_train['RGNTI'][self.df_train['RGNTI'] == i] = "Unknown"
      #self.df_train = self.df_train.drop(ids)
    
    ids = self.df_train.index[self.df_train['RGNTI'] == 'Unknown'].tolist()
    ids = ids[self.settings['MAX_TEXTS'][self.settings['LEVEL']]:]
    self.df_train = self.df_train.drop(ids)

    # Убираем тексты, в которых меньше текстов, чем задано в настройках
    self.df_train = self.df_train.groupby('RGNTI').filter(lambda x: len(x) > self.settings['MIN_TEXTS'][self.settings['LEVEL']])
  
  def splitText(self):
    # Проведем токенезацию, лемматизацию и удалим стоп слова
    self.df_train['text'] = self.df_train['text'].astype(str).swifter.progress_bar(enable=False, desc='split_train').apply(lambda x: x.split())
    self.df_test['text'] = self.df_test['text'].astype(str).swifter.progress_bar(enable=False, desc='split_test').apply(lambda x: x.split())
  
  def removeNotInRGNTI(self):
    compYtoRGNTI_ = set(self.df_train['RGNTI'].to_list())
    compYtoRGNTI_.add("Unknown")
    self.compYtoRGNTI = {}
    n = 0
    for i in compYtoRGNTI_:
      self.compYtoRGNTI[n] = i
      n += 1

    self.compRGNTItoY = dict((v,k) for k,v in self.compYtoRGNTI.items())

    self.df_test['RGNTI'][~self.df_test['RGNTI'].isin(compYtoRGNTI_)] = "Unknown"
    
    #self.df_test = self.df_test[self.df_test['RGNTI'].isin(compYtoRGNTI_)]
    return (self.compYtoRGNTI, self.compRGNTItoY)
  
  def prepareAll(self, sep='\t'):
    self.readData(sep)
    self.concatColumns()
    self.checkCorrect()
    self.chooseLevel()
    self.removeTexts()
    self.splitText()
    return self.removeNotInRGNTI()

  def toCat(self):
    y_train = self.df_train['RGNTI'].apply(lambda x: self.compRGNTItoY[x] if x in self.compRGNTItoY else 'Unknown')
    y_test = self.df_test['RGNTI'].apply(lambda x: self.compRGNTItoY[x] if x in self.compRGNTItoY else 'Unknown')
    return (y_train, y_test)

    
