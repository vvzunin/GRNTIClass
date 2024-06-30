from gensim.models import Word2Vec
from gensim.models import FastText
import numpy as np
from tqdm import tqdm
import os.path

class VectorizeWord2Vec():
  def __init__(self, settings, type='word2vec', path = '', workers=8):
    self.settings = settings
    if type == 'word2vec':
      self.model = Word2Vec(
        vector_size = settings['vector_size'],
        window = settings['window_size'],
        sg = 1 if settings['skip-gram'] else 0,
        hs = settings['hs'],
        negative = settings['negative'],
        ns_exponent = settings['ns_exponent'],
        cbow_mean = settings['cbow_mean'],
        alpha = settings['alpha'],
        min_alpha = settings['min_alpha'],
        sample = settings['sample'],        
        workers=workers
      )
    elif (type == 'fasttext'):
       self.model = FastText(
        vector_size = settings['vector_size'],
        window = settings['window_size'],
        sg = 1 if settings['skip-gram'] else 0,
        hs = settings['hs'],
        negative = settings['negative'],
        ns_exponent = settings['ns_exponent'],
        cbow_mean = settings['cbow_mean'],
        alpha = settings['alpha'],
        min_alpha = settings['min_alpha'],
        sample = settings['sample'],
        workers=workers
      )
    self.type = type
    self.path = path
  
  def build_vocab(self, vocab_text):
    self.model.build_vocab(vocab_text)

  def train(self, data, try_load = True, save = True):
    if not os.path.exists(self.path):
      os.makedirs(self.path)
    filename = self.type
    for i in self.settings:
      filename += '_' + '{}'.format(self.settings[i])
    filename += '.model'
    if try_load:
      try:
        self.model = Word2Vec.load(self.path + filename)
      except IOError as e:
        self.model.train(data, total_examples=self.model.corpus_count, epochs=self.settings['epochs'], report_delay=1, compute_loss = self.settings['compute_loss'],)
        if (save):
          self.model.save(self.path + filename)
    else:
      self.model.train(data, total_examples=self.model.corpus_count, epochs=self.settings['epochs'], report_delay=1 , compute_loss = self.settings['compute_loss'],)
      if (save):
        self.model.save(self.path + filename)


  def vectorizeWords(self, words):
    words_vecs = [self.model.wv[word] for word in words if word in self.model.wv]
    if len(words_vecs) == 0:
      return np.zeros(100)
    words_vecs = np.array(words_vecs)
    return words_vecs.mean(axis=0)
  
  def vectorizeAll(self, text):
    return np.array([self.vectorizeWords(sentence) for sentence in text])