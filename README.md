 <p align="center"> 
 <img src = "https://raw.githubusercontent.com/ARBML/tkseem/master/logo.png" width = "200px"/>
 </p>
 
**tkseem (تقسيم)** is a tokenization library that encapsulates different approaches for tokenization and preprocessing of Arabic text. 

## Documentation 
Please visit [readthedocs](https://tkseem.readthedocs.io/en/latest/index.html) for the full documentation. 

## Installation
```
pip install tkseem
```

## Usage 


### Tokenization
<p align="center"> 
 <img src = "https://raw.githubusercontent.com/ARBML/tkseem/master/logo_tokenizers.png"/>
 </p>

```python
import tkseem as tk
tokenizer = tk.WordTokenizer()
tokenizer.train('samples/data.txt')

tokenizer.tokenize("السلام عليكم")
tokenizer.encode("السلام عليكم")
tokenizer.decode([536, 829])
```

### Caching 
```python
tokenizer.tokenize(open('data/raw/train.txt').read(), use_cache = True)
```

### Save and Load
```python

import tkseem as tk

tokenizer = tk.WordTokenizer()
tokenizer.train('samples/data.txt')

# save the model
tokenizer.save_model('vocab.pl')

# load the model
tokenizer = tk.WordTokenizer()
tokenizer.load_model('vocab.pl')
```

### Model Agnostic
```python
import tkseem as tk
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
running_times['Disjoint'] = calc_time(tk.DisjointLetterTokenizer)
running_times['Char'] = calc_time(tk.CharacterTokenizer)
```
## Notebooks 
We show how to use `tkseem` to train some nlp models. 
<table class="tg">

  <tr>
    <th class="tg-yw4l"><b>Name</b></th>
    <th class="tg-yw4l"><b>Description</b></th>
    <th class="tg-yw4l"><b>Notebook</b></th>
  </tr>
  <tr>
    <td class="tg-yw4l">Demo</td>
    <td class="tg-yw4l">Explain the syntax of all tokenizers. </td>
    <td class="tg-yw4l"><a href="https://colab.research.google.com/github/ARBML/tkseem/blob/master/tasks/demo.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" >
    </a></td>
  </tr>
  <tr>
    <td class="tg-yw4l">Sentiment Classification</td>
    <td class="tg-yw4l"> WordTokenizer for processing sentences and then train a classifier for sentiment classification. </td>
    <td class="tg-yw4l"><a href="https://colab.research.google.com/github/ARBML/tkseem/blob/master/tasks/sentiment_analysis/Sentiment%20Analysis.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" >
    </a></td>

  </tr>

  <tr>
    <td class="tg-yw4l">Meter Classification</td>
    <td class="tg-yw4l">CharacterTokenizer for meter classification using bidirectional GRUs. </td>
    <td class="tg-yw4l"><a href="https://colab.research.google.com/github/ARBML/tkseem/blob/master/tasks/meter_classification/Poetry%20Classification.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg"  >
    </a></td>
  </tr>

  <tr>
    <td class="tg-yw4l">Translation</td>
    <td class="tg-yw4l">Seq-to-seq model with attention. </td>
    <td class="tg-yw4l"><a href="https://colab.research.google.com/github/ARBML/tkseem/blob/master/tasks/translation/translation.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg"  >
    </a></td>
  </tr>

  <tr>
    <td class="tg-yw4l">Question Answering</td>
    <td class="tg-yw4l">Sequence to Sequence Model </td>
    <td class="tg-yw4l"><a href="https://colab.research.google.com/github/ARBML/tkseem/blob/master/tasks/question_answering/Question_Answering.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg"  >
    </a></td>
  </tr>
<table>

## Citation
```
@misc{tkseem2020,
  author = {Zaid Alyafeai and Maged Saeed},
  title = {tkseem: A Tokenization Library for Arabic.},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ARBML/tkseem}}
}
```

## Contribution 
This is an open source project where we encourage contributions from the community. 

## License
[MIT](LICENSE) license. 
