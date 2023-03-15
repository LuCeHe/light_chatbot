import os, json, shutil, time

from tqdm import tqdm
import numpy as np
import pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

tf.compat.v1.enable_eager_execution()

from tensorflow.keras.callbacks import ModelCheckpoint

from GenericTools.keras_tools.esoteric_losses.advanced_losses import *
from GenericTools.keras_tools.esoteric_models.wizard_of_wikipedia import metrics_wow, switch_external_knowledge, \
    tf_ContextKnowledgeEncoder, tf_ContextKnowledgeDecoder
from GenericTools.keras_tools.esoteric_optimizers.optimizer_selection import get_optimizer
from GenericTools.stay_organized.VeryCustomSacred import CustomExperiment, ChooseGPU
from GenericTools.keras_tools.esoteric_callbacks import *
from GenericTools.keras_tools.plot_tools import plot_history
from GenericTools.stay_organized.utils import setReproducible, str2val
from GenericTools.keras_tools.esoteric_layers import AddLossLayer, FakeAddMetricsLayer
from GenericTools.keras_tools.esoteric_optimizers import AdaBelief
from GenericTools.keras_tools.esoteric_tasks.task_redirection import Task
from GenericTools.keras_tools.huggingface_tools import HF_ModelUpgrade
from GenericTools.keras_tools.esoteric_models.transformer import TransformerEncoder
from GenericTools.keras_tools.learning_rate_schedules import DummyConstantSchedule
from innocent_explorations.chatbot.model import LightModel

CDIR = os.path.dirname(os.path.realpath(__file__))
DATAPATH = os.path.abspath(os.path.join(CDIR, 'data'))
os.makedirs(DATAPATH, exist_ok=True)

ex = CustomExperiment('dialogue', base_dir=CDIR, seed=0)
models_dict = {
    'E2E': LightModel,
}


@ex.config
def config():
    maxlen = 256
    max_knowledge = None  # 32
    batch_size = 4
    task_name = 'light'  # 'wow'
    epochs = 1
    steps_per_epoch = 1
    stop_time = 500  # 72000 = 20h, 54000 = 15h
    seed = 5
    # 'E2E', 'E2E_AriEL', 'E2E_AriEL_wLM', 'E2E_AriEL_wLM_sigmoid', 'E2E_AriEL_wLM_enc', 'E2E_AriEL_wLM_enc_sigmoid'
    # E2E_AriEL_wLM_enc_layernorm
    model_name = 'E2E'

    # comments = 'encoder_maxlen:128-decoder_maxlen:12'
    comments = ''
    # tests = ['on_data', 'max', 'beam', 'evaluations', 'dialogue']
    tests = ['evaluations']
    # tests = []
    load_model_path = None
    mha_type = ''  # original_transformer spiking_attention
    # load_model_path = r'C:\Users\PlasticDiscobolus\work\ariel_tests\good_experiments\2021-12-09--06-27-40--8113-dialogue_\trained_models\model_weights.h5'



# longest dialogue: 23 utterances in train
# vocabulary original WoW network, with BPE, 34883 subwords. Mine 29999

load_m = lambda x, mask_value, num_classes: tf.keras.models.load_model(
    x, custom_objects=
    {
        'TransformerEncoder': TransformerEncoder, 'tf_ContextKnowledgeDecoder': tf_ContextKnowledgeDecoder,
        'sparse_f1_on_max': sparse_f1_on_max, 'sparse_perplexity': sparse_perplexity,
        # 'masked_perplexity': masked_sparse_perplexity(mask_value),
        # 'masked_f1_on_max': masked_f1_on_max(num_classes, mask_value),
        'AddLossLayer': AddLossLayer, 'AddMetricsLayer': FakeAddMetricsLayer,
        'DummyConstantSchedule': DummyConstantSchedule,
    }
)
# masked_sparse_perplexity(mask_value),
# masked_f1_on_max(num_classes, mask_value),


@ex.automain
def main(maxlen, batch_size, epochs, stop_time, steps_per_epoch, seed,
         model_name, _log, comments, tests, load_model_path, mha_type, task_name):
    ChooseGPU(None)
    time_start = time.perf_counter()
    exp_dir = os.path.join(CDIR, ex.observers[0].basedir)
    setReproducible(seed)

    other_dir = os.path.join(exp_dir, 'other_outputs')
    images_dir = os.path.join(exp_dir, 'images')

    path_best_model = os.path.join(exp_dir, 'trained_models', 'model_weights.h5')

    gen_train = Task(batch_size=batch_size, steps_per_epoch=steps_per_epoch, epochs=epochs, task_name=task_name,
                     data_split='train', maxlen=maxlen, string_config=comments, data_path=DATAPATH)
    gen_val = Task(batch_size=batch_size, steps_per_epoch=steps_per_epoch, epochs=epochs, task_name=task_name,
                   data_split='valid', maxlen=maxlen, string_config=comments, data_path=DATAPATH)

    vocab_size = gen_train.vocab_size
    history_path = other_dir + '/log.csv'
    # val_data = gen_val.__getitem__()
    callbacks = [
        LearningRateLogger(),
        ModelCheckpoint(path_best_model, monitor='val_loss', verbose=1, save_best_only=True, mode='min'),
        tf.keras.callbacks.CSVLogger(history_path),
        TimeStopping(stop_time, 1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    ]

    pad_idx = gen_train.pad_idx

    encoder_maxlen = str2val(comments, 'encoder_maxlen', int, default=maxlen, split_symbol='-')
    decoder_maxlen = str2val(comments, 'decoder_maxlen', int, default=maxlen, split_symbol='-')
    rate = str2val(comments, 'dropout', float, default=.1)
    model = models_dict[model_name](input_vocab_size=vocab_size, target_vocab_size=vocab_size, pad_idx=pad_idx,
                                    encoder_maxlen=encoder_maxlen, decoder_maxlen=decoder_maxlen, rate=rate,
                                    comments=mha_type + '_' + comments)

    model.summary()
    model.run_eagerly = True
    lr = str2val(comments, 'lr', float, default=.0005)
    optimizer = get_optimizer(
        'Adam', lr, lr_schedule='',
        # 'AdaBelief', lr, lr_schedule='',
        total_steps=gen_train.epochs, clipnorm=.1,
        warmup_steps=gen_train.steps_per_epoch
    )
    model.compile(optimizer=optimizer, loss=None)

    if load_model_path is None:
        model.fit(gen_train, epochs=gen_train.epochs, validation_data=gen_val, callbacks=callbacks)
        if gen_train.epochs > 1:
            model = load_m(path_best_model, mask_value=pad_idx, num_classes=vocab_size)

    else:
        model = load_m(load_model_path, mask_value=pad_idx, num_classes=vocab_size)

    model.compile(
        optimizer,
        None,  # masked_sparse_crossentropy(mask_value=pad_idx),
        metrics=metrics_wow(num_classes=vocab_size, mask_value=pad_idx)
    )

    save_model = str2val(comments, 'savemodel', bool, default=False)

    if os.path.exists(path_best_model) and not save_model:
        os.remove(path_best_model)

    # Re-evaluate the model
    pbar = tqdm(total=len(tests), desc='Tests')

    results = {}
    if 'evaluations' in tests:
        for data_split in ['valid', 'test', 'unseen']:
            gen = Task(batch_size=batch_size, steps_per_epoch=steps_per_epoch, epochs=epochs,
                       task_name=task_name, data_split=data_split, maxlen=maxlen,
                       string_config=comments, data_path=DATAPATH)

            evaluation = model.evaluate(gen, verbose=0)
            results.update(
                {k + '_' + data_split: v for k, v in
                 zip(model.metrics_names, evaluation)})

        pbar.update(1)
        # print(results)

    if gen_train.epochs > 0 and load_model_path is None:
        h = pd.read_csv(history_path)
        history_dict = {k: h[k].tolist() for k in h.columns.tolist()}
        plot_history(histories=history_dict, plot_filename=os.path.join(images_dir, 'history.png'),
                     epochs=gen_train.epochs)

    all_sentences = []

    base_input_batch = gen_val.__getitem__()[0]
    if 'on_data' in tests:
        tokenizer = gen_train.tokenizer

        all_sentences.append('\n\nTokenizer on data:')
        gen_val.on_epoch_end()
        batch = gen_val.data_generation()
        for sample in batch['input_targets']:
            decoded = tokenizer.decode(sample)
            all_sentences.append('\n' + decoded)

        input_batch = base_input_batch

        all_sentences.append(
            '\n\nTokenizer on max predictions:')

        prediction = model.predict(input_batch)
        max_prediction = tf.argmax(prediction, -1)
        for mdl_sample, tgt_sample in zip(max_prediction, input_batch[-1]):
            mdl_output = tokenizer.decode(mdl_sample)
            tgt_output = tokenizer.decode(tgt_sample)
            all_sentences.append('\nmodel output:  {}'.format(mdl_output))
            all_sentences.append('\ntarget output: {}'.format(tgt_output))
        pbar.update(1)

    if 'max' in tests:
        tokenizer = gen_train.tokenizer

        all_sentences.append('\n\nGenerations:')

        input_batch = base_input_batch
        # generated_sentence = input_batch[-1][:, 0][..., None]
        generated_sentence = np.repeat(np.array([gen_train.start_idx] * batch_size)[..., None], maxlen, -1)
        input_batch[-1] = generated_sentence
        for i in range(maxlen - 1):
            prediction = model.predict(input_batch)
            new_token = np.argmax(prediction[:, i], -1)
            generated_sentence[:, i + 1] = new_token
            input_batch[-1] = generated_sentence

        for sample in generated_sentence:
            generated_sentence = tokenizer.decode(sample)
            all_sentences.append('\nmodel generation:  {}'.format(generated_sentence))
        pbar.update(1)

    if 'beam' in tests:
        tokenizer = gen_train.tokenizer

        all_sentences.append(
            '\n\nGenerations, HuggingFace beam-search:')

        input_batch = base_input_batch
        hf_model = HF_ModelUpgrade(model, input_batch[:-1], gen_val.start_idx, pad_idx, pad_idx, vocab_size)

        generated_sentence = hf_model.generate(
            input_ids=tf.constant(input_batch[-1][:, 0][..., None]), num_beams=3, num_return_sequences=1,
            do_sample=False, max_length=decoder_maxlen, min_length=3, fixed_length_input=True
        )
        for sample in generated_sentence:
            generated_sentence = tokenizer.decode(sample)
            all_sentences.append('\nmodel generation:  {}'.format(generated_sentence))
        pbar.update(1)

    if 'dialogue' in tests:
        tokenizer = gen_train.tokenizer

        gen_val = Task(batch_size=batch_size, steps_per_epoch=steps_per_epoch, epochs=epochs, task_name=task_name,
                       data_split='valid_random_split', maxlen=maxlen, string_config=comments, data_path=DATAPATH,
                       shuffle=False)
        all_sentences.append(
            '\n\nDialogue, HuggingFace beam-search:')

        for i in [0, 1, 2, 3]:
            base_input_batch = gen_val.__getitem__(i)[0]
            input_batch = base_input_batch
            hf_model = HF_ModelUpgrade(model, input_batch[:-1], gen_val.start_idx, pad_idx, pad_idx,
                                       vocab_size)

            generated_sentence = hf_model.generate(
                input_ids=tf.constant(input_batch[-1][:, 0][..., None]), num_beams=3,
                num_return_sequences=1,
                do_sample=True, max_length=decoder_maxlen, min_length=3, fixed_length_input=True,
                top_p=.02
            )

            for sample, context in zip(generated_sentence, input_batch[0]):
                generated_sentence = tokenizer.decode(sample)
                context_sentence = tokenizer.decode(context)
                all_sentences.append('\ncontext:           {}'.format(context_sentence))
                all_sentences.append('\nmodel generation:  {}'.format(generated_sentence))
        pbar.update(1)

    pbar.close()
    text_path = os.path.join(exp_dir, 'text', 'sentences.txt')
    with open(text_path, 'w', encoding="utf-8") as f:
        for sentence in all_sentences:
            _log.info(sentence)
            f.write(sentence)


    time_elapsed = (time.perf_counter() - time_start)

    results['time_elapsed'] = time_elapsed
    results['n_params'] = model.count_params()
    results_filename = os.path.join(other_dir, 'results.json')
    json.dump(results, open(results_filename, "w"))
    _log.info('DONE!')
