# Against All Odds: Overcoming Typology, Script, and Language Confusion in Multilingual Embedding Inversion Attacks


## Steps:
### 1. Train and evaluate inversion models using modified scripts
- using `vec2text`, extended from [vec2text](https://github.com/jxmorris12/vec2text).
- instead of using pooler output, we use the average of first and last layer of transformers for multilingual embeddings.
  

### 2. Evaluations 
- gather evaluations from inversion models. 
- refer to the README in `evaluations`


### 3. Implement  Language Confusion and Analyses

- refer to the README in `language_confusion`

### Language confusion results and analyses
- `results`
- `plots`
