import re
import math

def generate_binary(n):
  # 2^(n-1)  2^n - 1 inclusive
  bin_arr = range(0, int(math.pow(2,n)))
  bin_arr = [bin(i)[2:] for i in bin_arr]

  # Prepending 0's to binary strings
  max_len = len(max(bin_arr, key=len))
  bin_arr = [i.zfill(max_len) for i in bin_arr]

  return bin_arr

def perumte(n, k):
    output = []
    end = 1<<n   #this is the integer with binary developpement 1 followed by k zeros
    for j in range(end): # iterate until end means getting all k - 0-1 combinations
        comb = bin(j)[2:].zfill(n)
        perm = [int(x) for x in comb]
        if sum(perm) == k:
            output.append(perm)
    return output

def remove_tashkeel(text):
    text = re.sub(r"[ًٌٍَََُِّْ]", "", text)
    return text

def normalize_data(text):
    # use a mapping dictionary 
    regex = re.compile("|".join(map(re.escape, self.norm_dict.keys())))
    text  = regex.sub(lambda match: self.norm_dict[match.group(0)], text)
    return text 

def remove_english_chars(text):
    return re.sub('[a-zA-Z]', '', text)

def remove_digits(text):
    return re.sub('[0-9]', '', text)

# https://github.com/google-research/bert/blob/master/tokenization.py
def is_punctuation(char):
    cp = ord(char)
    if cp == 1567:
        return True
    if cp in range(33, 48) or cp in range(58, 65) or cp in range(91, 97) or cp in range(123, 127):
        return True
    else:
        return False 

def remove_extra_spaces(text):
    text = re.sub(" +", " ", text)
    return text

def clean_data(text):
    # remove tashkeel and special chars
    text = remove_tashkeel(text)
    chars = set(text)
    all_puncts = [char for char in chars if is_punctuation(char)]
    all_puncts = ("").join(all_puncts)
    text = re.sub(r"[{all_puncts}]", "", text)
    text = remove_extra_spaces(text)
    return text 

def split_on_binary(self, word, binary):
        out = []
        sub = word[0]
        
        for i, char in enumerate(word[1:]):
            if binary[i]:
                out.append(sub)
                sub = "##"
            sub += char
        out.append(sub)
        
        return out