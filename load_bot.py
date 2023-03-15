import os, shutil, json, argparse, time, re
import numpy as np

from official import nlp
from official.nlp.modeling.ops import sampling_module
from official.nlp.modeling.ops import beam_search
from transformers import BlenderbotSmallTokenizer

from GenericTools.keras_tools.esoteric_tasks.task_redirection import Task
from GenericTools.language_tools.unpadding import pad_sequences
from GenericTools.stay_organized.utils import str2val
from innocent_explorations.chatbot.dialogue import load_m
from innocent_explorations.chatbot.light_dataset import interaction_partner_tags, interaction_self_tags

CDIR = os.path.dirname(os.path.realpath(__file__))
DATAPATH = os.path.abspath(os.path.join(CDIR, 'data'))
EXPERIMENTS = os.path.join(CDIR, 'experiments')
bot_zip = os.path.join(CDIR, 'good_experiments', '2022-09-28--15-56-49--6237-dialogue_.zip')
bot_path = bot_zip.replace('.zip', '').replace('good_experiments', 'experiments')

config_path = os.path.join(bot_path, '1', 'config.json')
path_best_model = os.path.join(bot_path, 'trained_models', 'model_weights.h5')
path_tflite = os.path.join(bot_path, 'trained_models', 'model_weights.tflite')


def _symbols_to_logits_fn(model, knowledge, args, onnx=False):
    """Calculates logits of the next tokens."""
    if onnx:
        def pad(x):
            x = [
                np.concatenate([
                    args.pad_idx * np.ones((1, args.maxlen - v.shape[1])), v], axis=1)
                for v in x]
            return x

        def generate(model, input_batch):
            input_batch = [p.astype(np.float32) for p in input_batch]

            onnx_predictions = model.run(
                [model.get_outputs()[0].name],
                {
                    k.name: v
                    for k, v in zip(model.get_inputs(), input_batch)
                }
            )
            return onnx_predictions[0]
    else:
        pad = lambda x: x
        generate = lambda model, input_batch: model.predict(input_batch, verbose=0)

    mask_symbols = np.array([12914, 12913, 2, 1])
    miin = -50  # np.amin(logits) - 10

    # 195 am, 279 sorry, 69 just, 14 i, 58 sure
    mask_randomly = np.array([195, 279, 69, 14])

    def symbols_to_logits_fn(ids, i, temp_cache):
        input_batch = [*knowledge, ids, ids]

        input_batch = pad(input_batch)
        logits = generate(model, input_batch)
        logits = logits[:, i]

        if ids[:, i][0] in [695, 1048]:
            logits[:, mask_symbols] = miin
        logits[:, mask_randomly] = logits[:, mask_randomly] if np.random.rand() > .8 else miin

        return logits, temp_cache

    return symbols_to_logits_fn


SamplingModule = sampling_module.SamplingModule

from official.nlp.modeling.ops import decoding_module


def new_continue_search(self, state):
    i = state[decoding_module.StateKeys.CUR_INDEX]
    stop_condition = tf.math.logical_and(
        tf.less(i, self.max_decode_length),
        tf.math.not_equal(self.eos_id, state['ALIVE_SEQ'][0, -1])
    )
    stop_condition = tf.math.logical_and(
        stop_condition, tf.math.not_equal(0, state['ALIVE_SEQ'][0, -1])
    )
    return stop_condition


SamplingModule._continue_search = new_continue_search


def generate(args, model, knowledge):
    generation_batch = args.generation_batch if not 'beam' in args.text_sampling_conf \
        else args.generation_batch * args.beam_size

    initial_ids = np.array([args.start_idx] * generation_batch)

    if args.text_sampling_conf in ['topk', 'greedy', 'topp']:
        if args.text_sampling_conf == 'topk':
            text_sampling_conf = dict(sample_temperature=1., top_k=100, enable_greedy=False)
        elif args.text_sampling_conf == 'greedy':
            text_sampling_conf = dict(enable_greedy=True)
        elif args.text_sampling_conf == 'topp':
            # GOOD: top_p=.3 sample_temperature=1.
            text_sampling_conf = dict(sample_temperature=1., top_p=.2, enable_greedy=False)
        else:
            raise NotImplementedError


        decoder = SamplingModule(
            length_normalization_fn=None,
            symbols_to_logits_fn=_symbols_to_logits_fn(model, knowledge, args, args.get_onnx_model),
            vocab_size=args.vocab_size,
            max_decode_length=args.generation_length,
            eos_id=args.end_idx,
            padded_decode=False,
            **text_sampling_conf
        )

        ids, _ = decoder.generate(
            initial_ids=initial_ids, initial_cache={})

    elif args.text_sampling_conf == 'beam':
        # FIXME: beam search not working
        ids, _ = beam_search.sequence_beam_search(
            symbols_to_logits_fn=_symbols_to_logits_fn(model, knowledge, args, args.get_onnx_model),
            initial_ids=initial_ids,
            initial_cache={},
            vocab_size=args.vocab_size,
            beam_size=args.beam_size,
            alpha=0.6,
            max_decode_length=args.generation_length,
            eos_id=args.end_idx,
            padded_decode=False,
        )
    else:
        raise NotImplementedError

    return ids


import tensorflow as tf


def pack_knowledge(args, knowledge):
    knowledge = [pad_sequences(v, value=args.pad_idx) for v in knowledge]

    padlen = max([v.shape[1] for v in knowledge])

    knowledge = [
        np.concatenate([
            args.pad_idx * np.ones((1, padlen - v.shape[1])), v], axis=1)[..., :args.encoder_maxlen]
        for v in knowledge
    ]

    return knowledge


def update_knowledge(tokenizer, pi, dhi, di, bot_sentence, human_reply, args):
    generated_sentence = bot_sentence.replace('__start__', '_self_say').replace('__end__', '').replace(
        '__null__', '')
    generated_sentence = generated_sentence \
        if '_self_act' in generated_sentence else generated_sentence + ' _self_act'
    generated_sentence = generated_sentence \
        if '_self_emote' in generated_sentence else generated_sentence + ' _self_emote'

    reply = f'_partner_say {human_reply} _partner_act _partner_emote'
    new_dh = f' {generated_sentence} {reply}'
    new_dh = re.sub(' +', ' ', new_dh).lstrip()

    # print(new_dh)
    ndhi = tokenizer([new_dh], return_tensors='tf')['input_ids'].numpy().tolist()
    # print(len(dhi[0]), len(ndhi[0]))
    dhi = [h + nh for h, nh in zip(dhi, ndhi)]
    # print(len(dhi[0]))

    knowledge = [pi, dhi, di]
    knowledge = pack_knowledge(args, knowledge)

    # if not return_dhi:
    #     return knowledge
    # else:
    return knowledge, dhi


def main(args):
    results = {}
    if not os.path.exists(bot_path):
        shutil.unpack_archive(bot_zip, bot_path)

    with open(config_path) as f:
        config = json.load(f)

    print(config)

    comments = config.get('comments')
    task_name = config.get('task_name')
    maxlen = config.get('maxlen')
    print('maxlen', maxlen)
    batch_size = config.get('batch_size')
    mha_type = config.get('mha_type')
    model_name = config.get('model_name')

    tokenizer = BlenderbotSmallTokenizer.from_pretrained('facebook/blenderbot-90M')

    args.start_idx = tokenizer.bos_token_id
    args.pad_idx = tokenizer.pad_token_id
    args.vocab_size = tokenizer.vocab_size
    args.end_idx = tokenizer.eos_token_id

    args.maxlen = maxlen
    args.encoder_maxlen = str2val(comments, 'encoder_maxlen', int, default=maxlen, split_symbol='-')

    if not args.get_onnx_model:
        model = load_m(path_best_model, mask_value=args.pad_idx, num_classes=args.vocab_size)
        model.summary()

    else:
        import onnxruntime as rt

        filepath_onnx = path_best_model.replace('h5', 'onnx')

        if not os.path.exists(filepath_onnx):
            import tf2onnx, onnx
            import tensorflow as tf
            model = load_m(path_best_model, mask_value=args.pad_idx, num_classes=args.vocab_size)
            model.summary()
            model = tf.keras.models.Model(model.input, model.layers[-3].output)

            input_signature = [tf.TensorSpec([args.generation_batch, maxlen], tf.float32, name=l.name)
                               for i, l in enumerate(model.inputs)]

            onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
            onnx.save_model(onnx_model, filepath_onnx)
        model = rt.InferenceSession((filepath_onnx))

    task_description = 'There is a guitar that nobody is playing, but you know how to play it. It is a beautiful day, and you' \
                       'are ready for adventure'
    persona = 'I am a mean peasant'
    persona = 'I am a sweet and happy mermaid'
    persona = 'I am a sweet and happy mermaid. I love to play guitar, I would spend the whole day doing it.'
    dialogue_history = 'Why are you so mean?!'

    add_to_dh = {
        t: l.replace(t, '').lstrip() if t in l else ''
        for t in interaction_partner_tags for l in ['_partner_say' + dialogue_history]
    }
    dialogue_history = ' '.join([f'{k} {v}' for k, v in add_to_dh.items()])
    dialogue_history = re.sub(' +', ' ', dialogue_history).lstrip()

    print(dialogue_history)
    pi = tokenizer([persona], return_tensors='tf')['input_ids'].numpy().tolist()
    di = tokenizer([task_description], return_tensors='tf')['input_ids'].numpy().tolist()
    dhi = tokenizer([dialogue_history], return_tensors='tf')['input_ids'].numpy().tolist()

    knowledge = [pi, dhi, di]
    knowledge = pack_knowledge(args, knowledge)

    conversation_length = 10
    times = []
    for _ in range(conversation_length):

        time_start = time.perf_counter()
        ids = generate(args, model, knowledge)

        for sample in ids:
            generated_sentence = tokenizer.decode(sample)
            # print('_self_act', 695, tokenizer([' _self_act ']))
            # print('_self_emote', 695, tokenizer(['_self_emote']))
            print(ids.numpy().tolist())
            print(generated_sentence)

        if args.interact:
            reply = input('you:')
            # reply = np.random.choice(['good to know', 'what?', 'but you did not say that last time'])
            # interaction_partner_tags = ['_partner_say', '_partner_act', '_partner_emote']
            knowledge, dhi = update_knowledge(tokenizer, pi, dhi, di, generated_sentence, reply, args)

        time_elapsed = (time.perf_counter() - time_start)
        times.append(time_elapsed)
    print(times)
    results.update(vars(args))
    return results


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comments", default='', type=str, help="String to activate extra behaviors")
    parser.add_argument("--text_sampling_conf", default='topp', type=str, help="Random seed")
    parser.add_argument("--beam_size", default=3, type=int, help="Random seed")
    parser.add_argument("--get_onnx_model", default=0, type=int, help="Make lighter version of the model")
    parser.add_argument("--generation_length", default=50, type=int, help="How long the sentence produced should be")
    parser.add_argument("--generation_batch", default=1, type=int, help="How many sentences should be produced")
    parser.add_argument("--interact", default=1, type=int, help="How many sentences should be produced")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_argparse()
    main(args)
