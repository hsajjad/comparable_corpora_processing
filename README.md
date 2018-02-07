## Pipeline to extract comparable corpus from comparable corpora

Current setup: assumed for every source sentence, a respectable size of candidate list is available.

Use comparable-corpora.ipynb to follow the pipeline to find the best candidates given an MT output and script to calcualte the F-score

### Methodology
* Get a parallel corpus of the language pair you are interested in
* Tokenize it and build bilingual word embedding using VecMap
* Extract source and target sentence embeddings
* For every source sentence, find the closet target sentences using the embegging and create a list of nbest candidates
* Build an MT system on the parallel corpus and use it to translate every source sentence to target sentence
* Calculate BLEU of translated source sentence with its nbest candidates
* Choose the candidate sentence with the best bleu above a certain threshold
* Other option is to build a binary classifier on the nbest candidate list to decide good and bad pairs given a source sentence

### To-do:
* Have relative paths with sample files of gold, trian, MT, etc.
* Make preprcessed data and MT model available
* Add filtering scripts to find candidate list using mulitlingual sentence embeddings

