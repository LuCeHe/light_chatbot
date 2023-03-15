import os, re
import pandas as pd
import numpy as np
from tqdm import tqdm

from transformers import BlenderbotSmallTokenizer

mname = 'facebook/blenderbot-90M'
tokenizer = BlenderbotSmallTokenizer.from_pretrained(mname)

# FIXME: make this code download autonomously the LIGHT dataset

CDIR = os.path.dirname(os.path.realpath(__file__))
DATAPATH = os.path.abspath(os.path.join(CDIR, '..', '..', 'data', 'light_dialogue'))
os.makedirs(DATAPATH, exist_ok=True)
ParlAI_path = os.path.join(os.path.dirname(__file__), '..', '..', 'ParlAI')

data_path = os.path.join(ParlAI_path, 'data', 'light_dialogue',
                         'tasknameTrue_settingTrue_objectsTrue_person_namesTrue_personaself_emoteall_speechall_actionall_affordancesTrue_repeatnone_cands20_current_self_outputall_clip_cands10000_speech_prefixTrue')
tags = ['_object_desc', '_partner_act', '_partner_emote', '_partner_name', '_partner_say', '_self_act', '_self_emote',
        '_self_name', '_self_persona', '_self_say', '_setting_desc', '_setting_name', '_task_speech']
interaction_self_tags = ['_self_say', '_self_act', '_self_emote']
interaction_partner_tags = ['_partner_say', '_partner_act', '_partner_emote']
label_candidates = 'label_candidates:'
data_splits = ['train', 'valid', 'test', 'unseen']


# ds = [d for d in os.listdir(data_path) if 'valid' in d]

def main():
    n_lines = -1
    mixer_tag = '##change##'

    if len(os.listdir(DATAPATH)) == 0:
        for split in data_splits:
            # for d in ds:
            d = f'speech_{split}.txt' if not split == 'unseen' else 'speech_test_unseen.txt'
            h5d = d.replace('txt', 'h5')
            h5path = os.path.join(DATAPATH, h5d)

            if not os.path.exists(h5path):
                path = os.path.join(data_path, d)
                tot_lines = sum(1 for line in open(path, 'r', encoding='utf-8'))
                tot_lines = n_lines if n_lines > 0 else tot_lines

                with open(path, 'r', encoding='utf-8') as f:
                    list_results = []
                    dialogue_history = None
                    persona = None
                    for i in tqdm(range(tot_lines), desc=d):

                        results = {}
                        line = f.readline().strip()
                        line = line.replace('\\n', ' \n ').replace('text:', '').replace('\t', '')
                        line = line[:line.index(label_candidates)]

                        tags = [c for c in line.split(' ') if c.count('_') >= 2]

                        if '_task_speech' in line:
                            dialogue_history = ''
                            answer = None

                            partner_name = ''.join(
                                [t for t in line.split('\n') if '_partner_name' in t]).replace('_partner_name ', '')
                            my_name = ''.join(
                                [t for t in line.split('\n') if '_self_name' in t]).replace('_self_name ', '')

                            line, answer = line.split('labels:')

                            add_to_dh = {
                                t: l.replace(t, '').lstrip() if t in l else ''
                                for t in interaction_partner_tags + interaction_self_tags for l in line.split('\n')
                            }
                            dialogue_history = ' '.join([f'{k} {v}' for k, v in add_to_dh.items()])

                            persona = ''.join([
                                t for t in line.split('\n') if '_self_persona' in t]).replace('_self_persona ', '')

                            task_text = []
                            for t in line.split('\n'):
                                new_t = re.sub(' +', ' ', t).lstrip().rstrip()
                                if not any([k in t for k in
                                            ['_self_name', '_self_persona', '_setting_name', '_partner_name'
                                             ] + interaction_partner_tags + interaction_self_tags]):
                                    if '_object_desc' in t:
                                        new_t = '_object_desc' + new_t[new_t.index(':') + 1:]

                                    if not new_t.endswith('.'):
                                        new_t += '.'

                                    task_text.append(new_t)

                            task_text = ''.join(task_text)

                            for tag in tags:
                                task_text = task_text.replace(tag + ' ', mixer_tag)

                            task_text = task_text.split(mixer_tag)
                            np.random.shuffle(task_text)
                            task_text = ' '.join([t.lstrip() for t in task_text]).lstrip()
                            task_text = re.sub(' +', ' ', task_text)
                            persona = re.sub(' +', ' ', persona).lstrip()  # .capitalize()
                            dialogue_history = re.sub(' +', ' ', dialogue_history).lstrip()

                        else:
                            new_line, answer = line.split('labels:')
                            new_line = '\n'.join([t for t in new_line.split('\n') if not '_self_say' in t])

                            add_to_dh = {
                                t: l.replace(t, '').lstrip() if t in l else ''
                                for t in interaction_partner_tags for l in new_line.split('\n')
                            }

                            add_to_a = {
                                t: l.replace(t, '').lstrip() if t in l else ''
                                for t in interaction_self_tags for l in new_line.split('\n') if not t == '_self_say'
                            }

                            history_add = ' '.join([f'{k} {v}' for k, v in add_to_dh.items()])
                            dialogue_history += ' ' + history_add
                            answer += ' ' + ' '.join([f'{k} {v}' for k, v in add_to_a.items()])
                            answer = re.sub(' +', ' ', answer).lstrip()

                        if not answer is None:
                            answer = answer.replace("\"", '')

                            pi = tokenizer([persona], return_tensors='tf')['input_ids'].numpy()[0].tolist()
                            di = tokenizer([task_text], return_tensors='tf')['input_ids'].numpy()[0].tolist()
                            dhi = tokenizer([dialogue_history], return_tensors='tf')['input_ids'].numpy()[0].tolist()
                            ai = tokenizer([answer], return_tensors='tf')['input_ids'].numpy()[0].tolist()

                            ai = [tokenizer.bos_token_id] + ai + [tokenizer.eos_token_id]

                            results.update(persona=pi, description=di, dialogue_history=dhi, answer=ai)
                            list_results.append(results)
                            dialogue_history += f'_self_say {answer}'

                df = pd.DataFrame.from_records(list_results)
                df.to_hdf(h5path, key='df', mode='w')


if __name__ == '__main__':
    main()
