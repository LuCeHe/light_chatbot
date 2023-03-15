from tensorflow.keras.layers import *

from GenericTools.keras_tools.esoteric_layers import AddLossLayer, AddMetricsLayer
from GenericTools.keras_tools.esoteric_layers.random_switch import RandomSwitch
from GenericTools.keras_tools.esoteric_losses.advanced_losses import *
from GenericTools.keras_tools.esoteric_models.transformer import TransformerEncoder as tf_TransformerEncoder, \
    scaled_dot_product_attention_original
from GenericTools.keras_tools.esoteric_models.transformer import TransformerDecoder as tf_TransformerDecoder
from GenericTools.keras_tools.esoteric_models.wizard_of_wikipedia import tf_ContextKnowledgeDecoder, metrics_wow

scaled_att = lambda x: scaled_dot_product_attention_original(*x)[0]


def LightModel(num_layers=5, d_model=256, num_heads=2, dff=512, input_vocab_size=int(5e4),
               target_vocab_size=int(5e4), encoder_maxlen=1024, decoder_maxlen=1024,
               rate=.1, pad_idx=0, comments='original_transformer'):

    transformer_encoder = tf_TransformerEncoder(
        num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding=encoder_maxlen, rate=rate,
        config=comments
    )

    transformer_decoder = tf_ContextKnowledgeDecoder(
        num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding=decoder_maxlen, rate=rate,
        config=comments
    )

    persona = Input((None,))
    dialogue_history = Input((None,))
    description = Input((None,))
    input_targets = Input((None,))
    output_targets = Input((None,))

    persona_code = transformer_encoder(persona)
    dhistory_code = transformer_encoder(dialogue_history)
    d_code = transformer_encoder(description)

    # code = Average()([persona_code, d_code, dhistory_code])
    if 'averagegrounding' in comments:
        code = Average()([persona_code, d_code, dhistory_code])
    else:
        code = Lambda(scaled_att)([persona_code, d_code, dhistory_code])

    logits = transformer_decoder([input_targets, [code, None, None]], output_type='embedding_projection')

    logits = AddLossLayer(loss=sparse_perplexity)([output_targets, logits])
    # logits = AddLossLayer(loss=sparsesmape)([output_targets, logits])
    logits = AddMetricsLayer(metrics=metrics_wow(num_classes=input_vocab_size, mask_value=pad_idx))(
        [output_targets, logits])

    model = tf.keras.models.Model([persona, dialogue_history, description, input_targets, output_targets], logits)
    return model
