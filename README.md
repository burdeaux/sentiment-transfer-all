# This is a modification from the paper [《Style Transformer: Unpaired Text Style Transfer without Disentangled Latent Representation》](https://arxiv.org/abs/1905.05621)



## Requirements

pytorch >= 0.4.0

torchtext >= 0.4.0

nltk

fasttext == 0.8.3

kenlm



## Usage

The hyperparameters for the Style Transformer can be found in ''main.py''.


You can adjust them in the Config class from the ''main.py''.

## Dataset

If you want to test the dataset with different length, we provide yelp_20_2 (< 20) and yelp_long (< 30) datasets.

## Training
If you want to run the model, use the command:

```shell
python main.py
```





To evaluation the model, we used Fasttext,  NLTK and KenLM toolkit to evaluate the style control, content preservation and fluency respectively. The evaluation related files for the Yelp dataset are placed in the ''evaluator'' folder. 

The fluency is not provided in this repository, if you wanna test the perplexity,pls train the language model.


