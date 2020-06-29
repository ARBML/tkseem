import re

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
    