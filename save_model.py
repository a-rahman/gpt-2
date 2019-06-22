#!/usr/bin/env python3

import fire
import json
import os
import time
import numpy as np
import tensorflow as tf

import model, sample, encoder

def save_model(
    model_name='345M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=5,
    temperature=1,
    top_k=0,
    text='What is this?'
):
    """
    Interactively run the model
    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    """
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None], name='input')
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)

        context_tokens = enc.encode(text)
        print(context_tokens)
        generated = 0
        t0 = time.time()
        # print(text)
        for _ in range(nsamples // batch_size):
            out = sess.run(output, feed_dict={
                context: [context_tokens for _ in range(batch_size)]
            })[:, len(context_tokens):]
            print(out)

            ### To save model as SavedModel ###
            export_dir = os.path.join('models', '0')
            builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
            input_tensor = sess.graph.get_tensor_by_name('input:0')
            output_tensor = sess.graph.get_tensor_by_name('sample_sequence/output/Exit_3:0')
            model_input = tf.saved_model.build_tensor_info(input_tensor)
            model_output = tf.saved_model.build_tensor_info(output_tensor)
            signature_definition = tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'input': model_input},
                outputs={'output': model_output},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
            builder.add_meta_graph_and_variables(sess, 
                                                [tf.saved_model.tag_constants.SERVING],
                                                signature_def_map={
                                                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                                                signature_definition})
            builder.save()
            
        # tensorflow_model_server --port=9000 --model_name=0 --model_base_path=$(pwd)/models

if __name__ == '__main__':
    fire.Fire(interact_model)