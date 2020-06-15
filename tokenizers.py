import io
from collections import defaultdict
import pyarabic.araby as araby
import re
from farasa.segmenter import FarasaSegmenter
from pathlib import Path 
import sentencepiece as spm

class BaseTokenizer:
    def __init__(self,  input_data = None,
                        unknown_token = "<UNK>", padding_token = "<PAD>",
                        apply_farasa = False, max_tokens = 10000,
                        segm_token = '+', apply_split = True, 
                        clean = False, normalize = False):
        
        self.segm_token = segm_token
        self.max_tokens = max_tokens
        self.unknown_token = unknown_token
        self.padding_token = padding_token
        self.apply_farasa = apply_farasa
        self.clean = clean
        self.normalize = normalize
        self.apply_split = apply_split

    def read_data(self, input_data):
        
        # take either file or raw text 
        if isinstance(input_data, io.IOBase):
            self.corpus = input_data.read()
        elif isinstance(input_data, str):
            self.corpus = input_data
        else:
            raise("Error type not recognized !")
            return 
        
        
        if self.apply_farasa:
            print("applying farasa segmentation")
            segmenter = FarasaSegmenter(interactive = True)
            self.corpus = segmenter.segment(self.corpus)
            self.corpus = re.sub(r"[+]", segm_token, self.corpus)
            if self.segm_token is None:
               self. segm_token = '+'
        
        # clean, normalize and then split 
        if self.clean:
            self.corpus = self._clean(self.corpus)

        if self.normalize:
            self.corpus = self._normalize(self.corpus)

        if self.apply_split:
            Path("data").mkdir(parents = True, exist_ok = True)
            self.train_text, self.valid_text, self.test_text = self._split_corpus()
            self._write_data("data/train.txt", self.train_text)
            self._write_data("data/valid.txt", self.valid_text)
            self._write_data("data/test.txt",  self.test_text)
            del self.train_text, self.valid_text, self.test_text
            del self.corpus

    def _write_data(self, path, data):
        open(path, "w").write(data)
   

    def _split_corpus(self):
        # the criteris is the number of tokens
        corpus_tokens = self.corpus.split()
        corpus_size = len(corpus_tokens)
        split_length = int(len(corpus_tokens) * .8)
        trainval_tokens, test_tokens = corpus_tokens[:split_length],corpus_tokens[split_length:]
        split_length = int(len(trainval_tokens) * .8)
        train_tokens,val_tokens = trainval_tokens[:split_length],trainval_tokens[split_length:]
        joiner = lambda tokens: ' '.join(tokens)
        train_text,val_text,test_text = joiner(train_tokens),joiner(val_tokens), joiner(test_tokens)
        return train_text, val_text, test_text
    
    def _get_tokens_frequency(self,preprocessed_text):
        tokens_frequency = defaultdict(int)
        for word in preprocessed_text.split(" "):
            tokens_frequency[word]+=1
        return dict(tokens_frequency)
    
      # you are ready to tokenize from the tokens frequency dictionary:)

    def _split_word(self,word, number_of_subwords): 
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
        filtered_groups_of_subwords = list(filter(lambda group : all(subword in self.clean_tokens_frequency.keys() for subword in group), groups_of_subwords))
        return filtered_groups_of_subwords
    
    def _normalize(self, text):
        # replace alef, ha and wow 
        text = re.sub(r"[ىأإآ]", "ا", text)
        text = re.sub(r"ة","ت", text)
        text = re.sub(r"ؤ", "و", text)
        text = re.sub(r"ئ", "ي", text)
        return text 

    # https://github.com/google-research/bert/blob/master/tokenization.py
    def _is_punctuation(self, char):
        
        if char == self.segm_token:
            return False 

        cp = ord(char)
        if cp == 1567:
            return True
        if cp in range(33, 48) or cp in range(58, 65) or cp in range(91, 97) or cp in range(123, 127):
            return True
        else:
            return False 

    def _clean(self, text):
        # remove tashkeel and special chars
        text = araby.strip_tashkeel(text)
        chars = set(text)
        all_puncts = [char for char in chars if self._is_punctuation(char)]
        all_puncts = ("").join(all_puncts)
        text = re.sub(r"[{all_puncts}]", "", text)
        return text 
 
    
class FrequencyTokenizer(BaseTokenizer):
    tokens_frequency = None 

    def train(self):
       # this preprocessing will produce connected words with hashtags from the first
       preprocessed_text = open('data/train.txt', 'r').read()
       # populate tokens frequency dictionary
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
       # get the clean tokens frequency from the tokens frequency
       self.clean_tokens_frequency = {k:v for k,v in self.tokens_frequency.items() 
                                if k is not self.unknown_token and k is not self.padding_token}
  
       print(self.tokens_frequency)
     
    def tokenize(self, text):
        assert self.tokens_frequency
        tokens = []
        output_tokens = []
        for word in text.split():
            if word in self.tokens_frequency.keys():
                output_tokens.append(word) #not sure we need this ? 
            else:
                for i in range(2,len(word)+1,1):
                    groups_of_valid_subwords = self._split_word(word,i)
                    if groups_of_valid_subwords:
                        break
                # in case the word is out of our vocabulary, we will replace it with a special keyword <UNKNOWN>?
                if len(groups_of_valid_subwords)==0:
                    output_tokens.append(self.unknown_token)
                else:
                    # sort these groups based on their frequency in our vocabulary. The total sum of the frequency of all subwords is the criteria here
                    # we may need to discuss this one here btw :>
                    sorted_groups_of_valid_subwords = sorted(groups_of_valid_subwords, key=lambda group: 
                                                                sum(self.tokens_frequency[subword] for subword in group))
                    
                    tokens += sorted_groups_of_valid_subwords[-1]
                    for token in tokens:
                        if ' '+token not in ' '+text:
                            output_tokens.append(str('##'+token))
                        else:
                            output_tokens.append(token)
        return output_tokens
    
    @property
    def _tokens_list(self):
        return list(self.tokens_frequency.keys())
    
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
        detokenized = ''.join(tokens).replace('','')
        return detokenized

class SentencePieceTokenizer(BaseTokenizer):

    def train(self, model_type= "bpe"):
        spm.SentencePieceTrainer.train(input= 'data/train.txt', model_prefix='m', vocab_size=self.max_tokens, model_type = model_type, character_coverage=1.0, normalization_rule_name='identity')
        self.sp = spm.SentencePieceProcessor(model_file='m.model')
    
    def tokenize(self, text):
        return self.sp.encode(text, out_type = str)

    def encode(self, text):
        return self.sp.encode(text, out_type = int)

    def decode(self, encoded):
        return self.sp.decode(encoded)

    def detokenize(self, tokens):
        return ''.join(tokens).replace('▁', ' ')



