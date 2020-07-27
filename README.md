**Tkseem (تقسيم)** is a tokenization library that encapsulates different approaches for tokenization and preprocessing of Arabic text. We provide different preprocessing, cleaning, normalization and tokenization algorithms for Arabic text. 

## Features
* Cleaning
* Preprocessing
* Normalization
* Segmentation
* Tokenization

## Installation
```
pip install tkseem
```

## Usage 

### Preprocessors

```python
import tkseem as tk
tokenizer = tk.WordTokenizer()
tokenizer.process_data('samples/data.txt', clean = True, segment = True, normalize = True)
```

### Tokenization
```python
import tkseem as tk
tokenizer = tk.WordTokenizer()
tokenizer.process_data('samples/data.txt')
tokenizer.train()

tokenizer.tokenize("السلام عليكم")
tokenizer.encode("السلام عليكم")
tokenizer.decode([536, 829])
```

### Large Files
```python
import tokenizers as tk

# initialize
tokenizer = tk.WordTokenizer()
tokenizer.process_data('samples/data.txt')

# training 
tokenizer.train(large_file = True)
```

### Caching 
```python
tokenizer.tokenize(open('data/raw/train.txt').read(), cache = True)
```

### Save and Load
```python

import tkseem as tk

tokenizer = tk.WordTokenizer()
tokenizer.process_data('samples/data.txt')
tokenizer.train()

# save the model
tokenizer.save_model('freq.pl')

# load the model
tokenizer = tk.WordTokenizer()
tokenizer.load_model('freq.pl')
```

### Model Agnostic
```python
import tokenizers as tk
import time 
import seaborn as sns
import pandas as pd

def calc_time(fun):
    start_time = time.time()
    fun().train()
    return time.time() - start_time

running_times = {}

running_times['Word'] = calc_time(tk.WordTokenizer)
running_times['SP'] = calc_time(tk.SentencePieceTokenizer)
running_times['Random'] = calc_time(tk.RandomTokenizer)
running_times['Auto'] = calc_time(tk.AutoTokenizer)
running_times['Disjoint'] = calc_time(tk.DisjointLetterTokenizer)
running_times['Char'] = calc_time(tk.DisjointLetterTokenizer)
```
## Contribution 
We encourage contributions to this repository. 

## License
[MIT](LICENSE) license. 
