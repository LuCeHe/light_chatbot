# How to use this repo

In order to download and format the dataset you have to run the line ```python light_dataset.py```.
To train a network you have to run ```python dialogue.py```. Once trained you can deploy 
it online with ```python deploy_anvil.py```. I've done some initial tests using the onnx
format to speed up the models in the code ```test_onnxx_speedup.py```, but the improvements 
were not impressive.


# The content of this repo

I wanted to have a chatbot with a fixed personality, that has to play a role
in a fantasy game. So, I thought of the [LIGHT](https://arxiv.org/pdf/1903.03094.pdf) dataset
which is actually a series of text datasets where there's different characters in different
scenarios, and they have to chit-chat, produce emojis, and actions to take.

The architecture that I use is very similar to the Transformer architecture, but the
encoder is used to encode various inputs. It's based on the architecture proposed
on the [WoW](https://arxiv.org/pdf/1811.01241.pdf) article, where they have a chatbot
that can attend to Wikipedia articles to give a knowledge-grounded reply. In the case
of this chatbot with fixed personality, the idea was to have the encoder attend to
the persona of the character, the context of the interaction, and the history of 
the dialogue, which would be the knowledge to ground the reply, and the decoder would
only attend to the reply to give. This solves a few problems. The first one is that
the persona, the context, and the history are all fixed, cannot be influenced with a clever
prompt. This is in contrast to cases where for example the persona is given as an initial prompt, 
making it easy to ask the chatbot to simply forget that persona. Moreover, by passing them
through the encoder, they can be easily precomputed, so, the encoder can ideally be evaluated
many fewer times, than if all the text describing the persona, the context, and the history
had to be passed to the decoder as one would do if the chatbot was GPT based.
Additionally, this type of model is probably more efficient than relying on a huge GPT4 model, that
has to be trained to know everything about the world.

# Tricks that I would apply in another iteration

- Non autoregressive decoder model to speed up generation, since ideally the decoder would need
to be evaluated only once to produce a sentence, instead of looping for
each word as one would normally do;
- Pretrain encoder of WoW architecture with BERT and decoder with GPT2, since that would drastically
enrich the language of the architecture, and then fine tune it with the LIGHT datasets;
- Distil on a smaller model, since often one achieves better language by training on bigger models 
and then distil, than training directly on a small model. Then the small model would be faster at 
inference time.