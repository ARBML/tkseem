import io
import re
import os
import sys
import mmap
import pickle
import numpy as np
from utils import *
from tqdm import tqdm
from pathlib import Path
import sentencepiece as spm
from collections import defaultdict, Counter
from farasa.segmenter import FarasaSegmenter

class BaseTokenizer:
    """
    Base Tokenizer that implements the basic functionalities of a tokenizer
    """
    def __init__(self,  input_data = None,
                        unknown_token = "<UNK>", padding_token = "<PAD>",
                        segment = False, max_tokens = 10000,
                        segm_token = '+', split = True, 
                        clean = False, normalize = False):
        
        self.segm_token = segm_token
        self.max_tokens = max_tokens
        self.unknown_token = unknown_token
        self.padding_token = padding_token
        self.segment = segment
        self.clean = clean
        self.normalize = normalize
        self.split = split
        self.norm_dict = pickle.load(open("normalization_dictionary.pl", 'rb'))
        
        if self.segment:
            text_trap = io.StringIO()
            sys.stdout = text_trap
            self.segmenter = FarasaSegmenter(interactive = True)
            sys.stdout = sys.__stdout__
            
    def read_data(self, file_path):
        """
        Read [Segment] [Clean] [Normalize] [Split]
        """
        with open(file_path, 'r') as f:
            print("Reading the data ...")
            self.corpus = f.read()
        
        if self.segment:
            print("Segmenting the data ...")
            self.corpus = self.segmenter.segment(self.corpus)
            self.corpus = re.sub(r"[+]", self.segm_token, self.corpus)
        
        if self.clean:
            print("Cleaning the data ...")
            self.corpus = clean_data(self.corpus)

        if self.normalize:
            print("Normalizing the data ...")
            self.corpus = normalize_data(self.corpus)

        if self.split:
            print("Splitting the data ...")
            Path("data/raw").mkdir(parents = True, exist_ok = True)
            self.train_text, self.valid_text, self.test_text = self._split_corpus()
            self._write_data("data/raw/train.txt", self.train_text)
            self._write_data("data/raw/valid.txt", self.valid_text)
            self._write_data("data/raw/test.txt",  self.test_text)
            del self.train_text, self.valid_text, self.test_text
            del self.corpus

    def _get_tokens_frequency_quickly(self, file_path):
        """
        Get the tokens frequency quickly using memory mapping
        """
        encoding = "utf8"
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
                m.read(0)
                txt = ""
                i = 0 
                size_to_read = int(1e9)
                freq = Counter([])
                pbar = tqdm(total=int(m.size()/size_to_read))
                while i < m.size():
                    cur_txt = ""
                    data = m.read(size_to_read)
                    i+= size_to_read
                    try:
                        cur_txt= data.decode(encoding)
                    except:
                        cur_txt= (data + m.read(1)).decode(encoding) 
                        i +=1 
                    freq.update(cur_txt.split(' '))
                    pbar.update(1)
        return freq

    def _write_data(self, path, data):
        """
        Write the string data to a path
        """
        open(path, "w").write(data)

    def _split_corpus(self):
        """
        Split the data into train, valid and test
        """
        split_length = int(len(self.corpus) * .8)
        trainval_text, test_text = self.corpus[:split_length], self.corpus[split_length:]
        split_length = int(len(trainval_text) * .8)
        train_text, val_text = trainval_text[:split_length], trainval_text[split_length:]
        return train_text, val_text, test_text
    
    def _get_tokens_frequency(self, text):
        """
        Get tokens frequency using a dictionary
        """
        tokens_frequency = defaultdict(int)
        for word in text.split(" "):
            tokens_frequency[word]+=1
        return dict(tokens_frequency)
    
    def _split_word(self,word, number_of_subwords): 
        """
        Split a word into a specific number of sub-words
        """
        assert number_of_subwords>1
        groups_of_subwords = [] 
        def _split(_word, _number_of_subwords): 
            groups = [] 
            if _number_of_subwords==1 or len(_word) == 1: 
                groups.append(["##"+_word]) 
            else: 
                for i in range(1, len(_word), 1): 
                    groups.extend((_word[:i], *group) for group in _split(_word[i:],_number_of_subwords-1) if len(group)==_number_of_subwords-1) 
                groups_of_subwords = groups
            return groups 
         
        groups_of_subwords = _split(word,number_of_subwords)
        # if any of the subwords is not the vocabulray, filter out the whole group
        #filtered_groups_of_subwords = list(filter(lambda group : all(subword in self.clean_tokens_frequency.keys() for subword in group), groups_of_subwords))
        #return filtered_groups_of_subwords
        return groups_of_subwords
    
    def encode(self, text):
        """
        Convert text to ids 
        """
        return NotImplementedError
    
    def decode(self, encoded):
        """
        Convert ids to ids
        """
        return NotImplementedError

    def tokenize(self, text):
        """
        Convert text to tokens
        """
        raise NotImplementedError

    def detokenize(self, tokens):
        """
        Convert tokens to text
        """
        raise NotImplementedError

    def encode_and_save(self):
        """
        Encode all the files then save as numpy
        """
        Path("data/encoded").mkdir(parents = True, exist_ok = True)
        for file_path in os.listdir('data/raw/'):
            ids = self.encode(open(f"data/raw/{file_path}", 'r').read())
            np.save(f'data/encoded/{file_path[:-4]}.npy', ids)

class FrequencyTokenizer(BaseTokenizer):
    tokens_frequency = None 

    def train(self):
       preprocessed_text = open('data/raw/train.txt', 'r').read()
       sorted_tokens_frequency = {
                   k:v for k,v in sorted(
                           self._get_tokens_frequency(preprocessed_text).items(),
                           key=lambda x: x[1],
                           reverse=True
                           )
                       }

       limited_tokens_frequency = dict()
       limited_tokens_frequency[self.unknown_token] = -1
       limited_tokens_frequency[self.padding_token] = -1
       limited_tokens_frequency.update({k:v for k,v in list(sorted_tokens_frequency.items())[:self.max_tokens]})
       self.tokens_frequency = limited_tokens_frequency
       self.clean_tokens_frequency = {k:v for k,v in self.tokens_frequency.items() 
                                if k is not self.unknown_token and k is not self.padding_token}
       
    def tokenize(self, text):
        assert self.tokens_frequency
        tokens = []
        output_tokens = []
        for word in text.split():
            if word in self.tokens_frequency.keys():
                output_tokens.append(word) 
            else:
                output_tokens.append(self.unknown_token)
        return output_tokens
    
    def _tokens_list(self):
        return list(self.tokens_frequency.keys())
    
    def decode(self, encoded):
        decoded = [self.tokens_list()[id] for id in encoded]
        return decoded
    
    def encode(self,text):
        tokens = self.tokenize(text)
        encoded = [self._tokens_list().index(token) for token in tokens]
        return encoded
    
    def get_tokenid(self,token):
        return self.tokens_list().index(token)
    
    def get_token_from_id(self,id):
        return self.tokens_list()[id]
    

    def detokenize(self, tokens):
        detokenized = ''.join(tokens).replace('','')
        return detokenized

class SentencePieceTokenizer(BaseTokenizer):

    def train(self, model_type= "bpe"):
        spm.SentencePieceTrainer.train(input= 'data/raw/train.txt', model_prefix='m', vocab_size=self.max_tokens, model_type = model_type, character_coverage=1.0, normalization_rule_name='identity')
        self.sp = spm.SentencePieceProcessor(model_file='m.model')
    
    def tokenize(self, text):
        return self.sp.encode(text, out_type = str)

    def encode(self, text):
        return self.sp.encode(text, out_type = int)

    def decode(self, encoded):
        return self.sp.decode(encoded)

    def detokenize(self, tokens):
        return ''.join(tokens).replace('▁', ' ')

class AutoTokenizer(BaseTokenizer):

    def __init__(self, vocab = 'vocab.pl'):
        print("loading vocab ...")
        self.vocab = pickle.load(open(vocab, 'rb'))
        super().__init__(self)

 
    def tokenize(self, text):
        assert self.vocab
        tokens = []
        output_tokens = []
        for word in text.split():
            if word in self.vocab.keys():
                output_tokens.append(word)
            else:
                for i in range(2,len(word)+1,1):
                    groups_of_valid_subwords = self._split_word(word,i)
                    if groups_of_valid_subwords:
                        break
                if len(groups_of_valid_subwords)==0:
                    output_tokens.append(self.unknown_token)
                else:
                    sorted_groups_of_valid_subwords = sorted(groups_of_valid_subwords, key=lambda group: sum(self.vocab[subword] for subword in group))
                    
                    tokens = sorted_groups_of_valid_subwords[-1]
                    for token in tokens:
                        output_tokens.append(str(token))
        return output_tokens
    
    def _tokens_list(self):
        return list(self.vocab.keys())
    
    def decode(self, encoded):
        decoded = [self.tokens_list[id] for id in encoded]
        return decoded
    
    def encode(self,text):
        tokens = self.tokenize(text)
        encoded = [self._tokens_list.index(token) for token in tokens]
        return encoded
    
    def get_tokenid(self,token):
        return self.tokens_list.index(token)
    
    def get_token_from_id(self,id):
        return self.tokens_list[id]
    
    def detokenize(self, tokens):
        detokenized = ''.join(tokens).replace('##','')
        return detokenized

