# !pip install transformers
# !pip install sentencepiece

import transformers

from transformers import pipeline

generator = pipeline('text-generation', model= 'distilgpt2')

generator(
    'Today I am going to showcase our platform where',
    max_length= 100,
    num_return_sequences=4,
)

