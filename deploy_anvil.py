import os
import anvil.server
import numpy as np

from transformers import BlenderbotSmallTokenizer, BlenderbotTokenizer, BlenderbotForConditionalGeneration

from innocent_explorations.chatbot.dialogue import load_m
from innocent_explorations.chatbot.load_bot import generate, update_knowledge

CDIR = os.path.dirname(os.path.realpath(__file__))
DATAPATH = os.path.abspath(os.path.join(CDIR, '..', '..', 'data', 'chatbot'))

path_best_model = os.path.join(DATAPATH, 'model_weights.h5')

anvil.server.connect('LKVRXTKDP7N7A5EJHGUHESWK-XXMJTCRVTPECSYWD')

tokenizer_small = BlenderbotSmallTokenizer.from_pretrained('facebook/blenderbot-90M')
tokenizer_big = BlenderbotTokenizer.from_pretrained('facebook/blenderbot-400M-distill')
blenderbot = BlenderbotForConditionalGeneration.from_pretrained('facebook/blenderbot-400M-distill')

from transformers import pipeline

# summarizer = pipeline("summarization")
summary_signal = 'TL;DR'


args = lambda x: x
args.generation_batch = 1
args.start_idx = tokenizer_small.bos_token_id
args.pad_idx = tokenizer_small.pad_token_id
args.text_sampling_conf = 'topp'
args.get_onnx_model = 0
args.vocab_size = tokenizer_small.vocab_size
args.beam_size = 1
args.generation_length = 50
args.end_idx = tokenizer_small.eos_token_id
args.maxlen = 256
args.encoder_maxlen = args.maxlen

model = load_m(path_best_model, mask_value=args.pad_idx, num_classes=args.vocab_size)


# import tensorflowjs as tfjs
#
# tfjs_target_dir = path_best_model.replace()
# tfjs.converters.save_keras_model(model, DATAPATH)


@anvil.server.callable
def tokenize_text(text):
    return tokenizer_small([text], return_tensors='tf')['input_ids'].numpy().tolist()


@anvil.server.callable
def answer_questions(dhi, pi, di, human_reply, last_bot_sentence):
    knowledge, dhi = update_knowledge(tokenizer_small, pi, dhi, di, last_bot_sentence, human_reply, args)
    if np.random.rand() > -.5:
        ids = generate(args, model, knowledge)[0]
        generated_sentence = tokenizer_small.decode(ids)
    else:
        print(di)
        d = tokenizer_small.decode(di[0])
        p = tokenizer_small.decode(pi[0])
        dh = tokenizer_small.decode(dhi[0])
        print(d)
        print(p)
        print(dh)
        print(len(d), len(p), len(dh))

        # summary = summarizer(dh)[0]['summary_text']

        text_len = len(dh)
        generator = pipeline('text-generation', model='gpt2-medium', max_length=text_len + 30)
        generation = generator(dh + summary_signal)[0]['generated_text']
        summary = generation.split(summary_signal)[1]

        print(summary)
        inputs = tokenizer_big([f'{d} {summary} {p}'], return_tensors='pt')
        ids = blenderbot.generate(**inputs)
        generated_sentence = tokenizer_big.decode(ids)
        print(generated_sentence)
        print(f'{d} {dh} {p}')

    return generated_sentence, dhi


anvil.server.wait_forever()
