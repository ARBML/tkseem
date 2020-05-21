import io
from collections import defaultdict

class BaseTokenizer:
    corpus = None
    train_text = None
    test_text = None
    validation_text = None
    train_test_split = None
    train_val_split = None
    max_tokens = None
    tokens_frequency = None
    unkown_token = None
    padding_token = None
    #this tokens does not have the hash (#) symbol. If I am not mistaken, we will use the hash symbole only when detokenizing
    # we will use this clean one for the purpose of tokenizing.
    clean_tokens_frequency = None
    def __init__(self, textfile=None,
                        unknown_token = "<UNK>", padding_token="<PAD>",
                        train_test_split = 0.8, train_val_split=0.2,
                        max_tokens=10000,):
        assert isinstance(textfile,io.IOBase)
        # text has higher priority than textfile :)
        self.corpus = textfile.read()
        self.train_text,self.validation_text, self.test_text = self._split_corpus()
        self.max_tokens = max_tokens
        self.unknow_token = unknown_token
        self.padding_token = padding_token
        self.train_test_split = train_test_split
        self.train_val_split = train_val_split
        print("training the model on the given corups...")
        self._train()
        print("done with model training on the corpus")
    
    def _split_corpus(self):
        # the criteris is the number of tokens
        corpus_tokens = self.corpus.split()
        corpus_size = len(corpus_tokens)
        trainval_tokens,test_tokens = corpus_tokens[:self.train_test_split*corpus_size],corpus_tokens[self.train_test_split*corpus_size:]
        train_tokens,val_tokens = trainval_tokens[:self.train_val_split*len(trainval_tokens)],trainval_tokens[self.train_val_split*len(trainval_tokens):]
        joiner = lambda tokens: ' '.join(tokens)
        train_text,val_text,test_text = joiner(train_tokens),joiner(val_tokens), joiner(test_tokens)
        return train_text, val_text, test_text

    
    def _preprocess(self,text):
        preprocessed_text = text.replace("## ", " ")
        preprocessed_text = preprocessed_text.replace("##", " ##")
        return preprocessed_text

    def _get_tokens_frequency(self,preprocessed_text):
        tokens_frequency = defaultdict(int)
        for word in preprocessed_text.split(" "):
            tokens_frequency[word]+=1
        return dict(tokens_frequency)
    
    def _train(self):
        # first preprocess
        # this preprocessing will produce connected words with hashtags from the first
        preprocessed_text = self._preprocess(self.train_text)
        # populate tokens frequency dictionary
        sorted_tokens_frequency = {
                    k:v for k,v in sorted(
                            self._get_tokens_frequency(preprocessed_text).items(),
                            key=lambda x: x[1],
                            reverse=True
                            )
                        }

        limited_tokens_frequency = dict()
        limited_tokens_frequency[self.unkown_token] = -1
        limited_tokens_frequency[self.padding_token] = -1
        # include only max_tokon tokens in the dict
        limited_tokens_frequency.update({k:v for k,v in sorted_tokens_frequency.items()[:self.max_tokens]})
        self.tokens_frequency = limited_tokens_frequency
        # get the clean tokens frequency from the tokens frequency
        self.clean_tokens_frequency = {k.replace('#',''):v for k,v in self.tokens_frequency.items() 
                                            if k is not self.unknow_token and k is not self.padding_token}
        # you are ready to tokenize from the tokens frequency dictionary:)

    def _split_word(self,word, number_of_subwords): 
        assert number_of_subwords>1
        # assert number_of_subwords>0
        # if len(word) == number_of_subwords:
        #     return [word] if word in self.tokens_frequency.keys() else []
        groups_of_subwords = [] 

        def _split(_word, _number_of_subwords): 
            groups = [] 
            if _number_of_subwords==1 or len(_word) == 1: 
                groups.append([_word]) 
            else: 
                for i in range(1, len(_word), 1): 
                    groups.extend((_word[:i],*group) for group in _split(_word[i:],_number_of_subwords-1) if len(group)==_number_of_subwords-1) 
            return groups 
         
        groups_of_subwords = _split(word,number_of_subwords)
        # if any of the subwords is not the vocabulray, filter out the whole group
        filtered_groups_of_subwords = list(filter(lambda group : all(subword in self.clean_tokens_frequency.keys() for subword in group),
                                            groups_of_subwords))
        return filtered_groups_of_subwords


    def tokenize(self, text):
        assert len(self.clean_tokens_frequency)
        tokens = []
        for word in text.split():
            if word in self.clean_tokens_frequency.keys():
                tokens.append(word)
            else:
                for i in range(2,len(word)+1,1):
                    groups_of_valid_subwords = self._split_word(word,i)
                    if groups_of_valid_subwords:
                        break
                # in case the word is out of our vocabulary, we will replace it with a special keyword <UNKNOWN>?
                if len(groups_of_valid_subwords)==0:
                    tokens.append(self.unkown_token)
                else:
                    # sort these groups based on their frequency in our vocabulary. The total sum of the frequency of all subwords is the criteria here
                    # we may need to discuss this one here btw :>
                    sorted_groups_of_valid_subwords = sorted(groups_of_valid_subwords, key=lambda group: 
                                                                sum(self.clean_tokens_frequency[subword] for subword in group))
                    
                    tokens += sorted_groups_of_valid_subwords[-1]
                    hashed_tokens = []
                    for token in tokens:
                        if ' '+token not in ' '+text:
                            hashed_tokens.append(str('##'+token))
                        else:
                            hashed_tokens.append(token)
        return hashed_tokens
    
    @property
    def tokens_list(self):
        return list(self.tokens_frequency.keys())
    
    def decode(self, encoded):
        assert all(float(elem) for elem in encoded)
        decoded = [self.tokens_list[id] for id in encoded]
        return decoded
    
    def encode(self,text):
        encoded = [self.tokens_list.index(token) for token in text.split()]
        return encoded
    
    def get_tokenid(self,token):
        return self.tokens_list.index(token)
    
    def get_token_from_id(self,id):
        return self.tokens_list[id]
    

    def detokenize(self,text):
        detokenized = text.replace(' ##','')
        return detokenized


'''
TODO:
-----------------------
fixed dict size
divide to train,test and validate
save/load model
save as numpy()
-----------------------
'''