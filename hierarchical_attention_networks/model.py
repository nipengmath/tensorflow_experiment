# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from func import cudnn_gru, native_gru, dot_attention, summ, dropout, ptr_net, dense, bi_shortcut_stacked_lstm_return_sequences


class Model(object):
    def __init__(self, config, batch, word_mat=None, trainable=True, opt=True):
        self.config = config
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        self.c, self.y, self.id = batch.get_next()
        self.is_train = tf.get_variable(
            "is_train", shape=[], dtype=tf.bool, trainable=False)
        self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(
            word_mat, dtype=tf.float32), trainable=True)

        self.num_classes = config.num_classes

        self.decay_steps, self.decay_rate = config.decay_steps, config.decay_rate
        self.clip_gradients = config.clip_gradients
        self.hidden_size = config.hidden_size
        self.keep_prob = config.keep_prob

        # if opt:
        #     N, CL = config.batch_size, config.char_limit
        #     self.c_maxlen = tf.reduce_max(self.c_len)
        #     self.q_maxlen = tf.reduce_max(self.q_len)
        #     self.c = tf.slice(self.c, [0, 0], [N, self.c_maxlen])
        #     self.q = tf.slice(self.q, [0, 0], [N, self.q_maxlen])

        #     self.c_mask = tf.slice(self.c_mask, [0, 0], [N, self.c_maxlen])
        #     self.q_mask = tf.slice(self.q_mask, [0, 0], [N, self.q_maxlen])
        #     self.ch = tf.slice(self.ch, [0, 0], [N, CL])
        #     self.qh = tf.slice(self.qh, [0, 0], [N, CL])
        # else:

        self.c_maxnum, self.c_maxlen = config.para_max_num, config.para_max_length

        self.ready()

        self.all_params = tf.trainable_variables()

        if trainable:
            self.lr = tf.get_variable(
                "lr", shape=[], dtype=tf.float32, trainable=False)
            self.opt = tf.train.AdadeltaOptimizer(
                learning_rate=self.lr, epsilon=1e-6)
            grads = self.opt.compute_gradients(self.loss)
            gradients, variables = zip(*grads)
            capped_grads, _ = tf.clip_by_global_norm(
                gradients, config.grad_clip)
            self.train_op = self.opt.apply_gradients(
                zip(capped_grads, variables), global_step=self.global_step)

    def ready(self):
        config = self.config
        N, PN, PL, HS = config.batch_size, self.c_maxnum, self.c_maxlen, self.hidden_size
        gru = native_gru

        print("c", self.c)
        self.c_mask = tf.cast(tf.reshape(self.c, [N*PN, PL]), tf.bool)
        print("c_mask", self.c_mask)
        self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
        print("c_len", self.c_len)

        with tf.variable_scope("emb"):
            c_emb = tf.nn.embedding_lookup(self.word_mat, self.c)
            print("c_emb", c_emb)
            c_emb = tf.reshape(c_emb, [N*PN, PL, c_emb.get_shape().as_list()[-1]])
            print("c_emb reshape", c_emb)

        with tf.variable_scope("word_encoder"):
            rnn = gru(num_layers=1, num_units=HS, batch_size=N*PN, input_size=c_emb.get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train, scope="word")
            c_encoder = rnn(c_emb, seq_len=self.c_len)
            print("c_encoder", c_encoder)

        with tf.variable_scope("word_attention"):
            dim = c_encoder.get_shape().as_list()[-1]
            u = tf.nn.tanh(dense(c_encoder, dim, use_bias=True, scope="dense"))
            print("u", u)
            u2 = tf.reshape(u, [N*PN*PL, dim])
            print("u2", u2)
            uw = tf.get_variable("uw", [dim, 1])
            alpha = tf.matmul(u2, uw)
            print("alpha", alpha)
            alpha = tf.reshape(alpha, [N*PN, PL])
            print("alpha", alpha)
            alpha = tf.nn.softmax(alpha, axis=1)
            print("alpha", alpha)
            s = tf.matmul(tf.expand_dims(alpha, axis=1), c_encoder)
            print("s", s)

        with tf.variable_scope("sent_encoder"):
            dim = s.get_shape().as_list()[-1]
            s = tf.reshape(s, [N, PN, dim])
            print("s", s)

            #s_mask = tf.cast(tf.reshape(s[:,:,0], [N, PN]), tf.bool)
            #print("s_mask", s_mask)
            #s_len = tf.reduce_sum(tf.cast(s_mask, tf.int32), axis=1)
            #print("s_len", s_len)
            s_len = tf.constant([PN for _ in range(N)],shape=[N,], name='s_len')
            print("s_len", s_len)
            rnn = gru(num_layers=1, num_units=HS, batch_size=N, input_size=dim,
                      keep_prob=config.keep_prob, is_train=self.is_train, scope="sent")
            h = rnn(s, seq_len=s_len)
            print("h", s)

        with tf.variable_scope("sent_attention"):
            dim = s.get_shape().as_list()[-1]
            u = tf.nn.tanh(dense(s, dim, use_bias=True, scope="dense"))
            print("u", u)
            u2 = tf.reshape(u, [N*PN, dim])
            print("u2", u2)
            us = tf.get_variable("us", [dim, 1])
            print("us", us)
            alpha2 = tf.matmul(u2, us)
            print("alpha2", alpha2)
            alpha2 = tf.reshape(alpha2, [N, PN])
            print("alpha2", alpha2)
            alpha2 = tf.nn.softmax(alpha2, axis=1)
            print("alpha2", alpha2)
            print(tf.expand_dims(alpha2, axis=1))
            print(s)
            v = tf.matmul(tf.expand_dims(alpha2, axis=1), s)
            print("v", v)
            v = tf.reshape(v, [N, dim])
            print("v", v)

        with tf.variable_scope("output"):
            self.logits = dense(v, self.num_classes,
                                use_bias=True, scope="output")
            print("logits", self.logits)
            self.scores = tf.nn.softmax(self.logits)
            print("scores", self.scores)

        with tf.variable_scope("predict"):
            print(self.y)
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.logits,
                labels=tf.stop_gradient(self.y))
            self.loss = tf.reduce_mean(losses)
            print("loss", self.loss)

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step

    def compute_acc_prf(self):
        ## accuracy
        correct_predictions = tf.equal(self.predictions, tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        ## PRF
        predictions = tf.argmax(self.logits, 1)
        actuals = tf.argmax(self.y, 1)
        ones_like_actuals = tf.ones_like(actuals)
        zeros_like_actuals = tf.zeros_like(actuals)
        ones_like_predictions = tf.ones_like(predictions)
        zeros_like_predictions = tf.zeros_like(predictions)

        self.tp_op = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(actuals, ones_like_actuals),
                    tf.equal(predictions, ones_like_predictions)
                ),
                "float"
            )
        )

        self.tn_op = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(actuals, zeros_like_actuals),
                    tf.equal(predictions, zeros_like_predictions)
                ),
                "float"
            )
        )

        self.fp_op = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(actuals, zeros_like_actuals),
                    tf.equal(predictions, ones_like_predictions)
                ),
                "float"
            )
        )

        self.fn_op = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(actuals, ones_like_actuals),
                    tf.equal(predictions, zeros_like_predictions)
                ),
                "float"
            )
        )
