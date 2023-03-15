import os, tf2onnx, onnx, time

import onnxruntime as rt
import numpy as np

import tensorflow as tf

from transformers import BlenderbotSmallTokenizer

from innocent_explorations.chatbot.light_dataset import interaction_partner_tags
from innocent_explorations.chatbot.load_bot import pack_knowledge
from innocent_explorations.chatbot.model import LightModel

CDIR = os.path.dirname(os.path.realpath(__file__))
EXPERIMENTS = os.path.join(CDIR, 'experiments')

filepath_h5 = os.path.join(EXPERIMENTS, 'example_model.h5')
filepath_onnx = os.path.join(EXPERIMENTS, 'example_model.onnx')

batch_size, time_steps = 1, 256
n_tries = 20

model = LightModel()
model = tf.keras.models.Model(model.input, model.layers[-3].output)

if not os.path.exists(filepath_h5):
    model.save(filepath_h5, overwrite=True)

if not os.path.exists(filepath_onnx):
    input_signature = [tf.TensorSpec([batch_size, None], tf.float32, name=l.name) for l in model.inputs]

    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
    onnx.save_model(onnx_model, filepath_onnx)

else:
    onnx_model = onnx.load(filepath_onnx)

mname = 'facebook/blenderbot-90M'
tokenizer = BlenderbotSmallTokenizer.from_pretrained(mname)

task_description = 'There is a guitar that nobody is playing, but you know how to play it. It is a beautiful day, and you' \
                   'are ready for adventure'
persona = 'I am a mean peasant'
dialogue_history = 'Why are you so mean?!'

add_to_dh = {
    t: l.replace(t, '').lstrip() if t in l else ''
    for t in interaction_partner_tags for l in ['_partner_say' + dialogue_history]
}
dialogue_history = ' '.join([f'{k} {v}' for k, v in add_to_dh.items()])

print(dialogue_history)
pi = tokenizer([persona], return_tensors='tf')['input_ids'].numpy().tolist()
di = tokenizer([task_description], return_tensors='tf')['input_ids'].numpy().tolist()
dhi = tokenizer([dialogue_history], return_tensors='tf')['input_ids'].numpy().tolist()

knowledge = [pi, dhi, di]

args = lambda x: x
args.pad_idx = 0
args.encoder_maxlen = time_steps

knowledge = pack_knowledge(args, knowledge)
ids = tf.ones_like(knowledge[0]).numpy()
print(ids)
input_tensors = [*knowledge, ids, ids]
input_tensors = [tf.keras.preprocessing.sequence.pad_sequences(v, value=args.pad_idx, maxlen=time_steps)
                 for v in input_tensors]
input_tensors = [p.astype(np.float32) for p in input_tensors]

print([p.shape for p in input_tensors])

session = rt.InferenceSession((filepath_onnx))
# onnx_predictions = session.run([label_name], {input_name: x_test.astype(np.float32)})[0]

onnx_times = []
for _ in range(n_tries):
    start_time = time.time()
    onnx_predictions = session.run(
        [session.get_outputs()[0].name],
        {
            k.name: v
            for k, v in zip(session.get_inputs(), input_tensors)
        }
    )
    onnx_times.append(time.time() - start_time)
print("Time taken by ONNX: ", np.mean(onnx_times))
print(onnx_times)

tf_times = []
for _ in range(n_tries):
    start_time = time.time()

    prediction = model.predict(input_tensors)
    tf_times.append(time.time() - start_time)
print("Time taken by TF: ", np.mean(tf_times))
print(tf_times)

print('DONE')
