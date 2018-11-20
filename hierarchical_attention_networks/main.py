# -*- coding: utf-8 -*-

import tensorflow as tf
import json
from sklearn.externals import joblib
#import ujson as json
import numpy as np
import os
from collections import defaultdict
from datetime import datetime
from sklearn import linear_model, svm

from model import Model
from util import get_record_parser, get_batch_dataset, get_dataset


def train(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.train_eval_file, "r") as fh:
        train_eval_file = json.load(fh)
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)
    with open(config.dev_meta, "r") as fh:
        meta = json.load(fh)

    dev_total = meta["total"]

    print("Building model...")
    parser = get_record_parser(config)
    train_dataset = get_batch_dataset(config.train_record_file, parser, config)

    dev_dataset = get_dataset(config.dev_record_file, parser, config)
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)
    train_iterator = train_dataset.make_one_shot_iterator()
    dev_iterator = dev_dataset.make_one_shot_iterator()

    model = Model(config, iterator, word_mat)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    loss_save = 100.0
    save_f1 = 0.1
    patience = 0
    lr = config.init_lr

    with tf.Session(config=sess_config) as sess:
        param_num = sum([np.prod(sess.run(tf.shape(v))) for v in model.all_params])
        print('There are {} parameters in the model'.format(param_num))

        writer = tf.summary.FileWriter(config.log_dir)
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter('./log/log', sess.graph)
        saver = tf.train.Saver(max_to_keep=10000)

        train_handle = sess.run(train_iterator.string_handle())
        dev_handle = sess.run(dev_iterator.string_handle())
        sess.run(tf.assign(model.is_train, tf.constant(True, dtype=tf.bool)))
        sess.run(tf.assign(model.lr, tf.constant(lr, dtype=tf.float32)))


        # ## 加载训练好的模型
        # if os.path.exists(config.save_dir + "/checkpoint"):
        #     print("Restoring Variables from Checkpoint.")
        #     saver.restore(sess,tf.train.latest_checkpoint(config.save_dir))
        #     ## 设置学习率
        #     ## sess.run(tf.assign(model.lr, tf.constant(0.0001, dtype=tf.float32)))

        for idx in range(1, config.num_steps + 1):
            global_step = sess.run(model.global_step) + 1
            ## print("=== %s  %s ===" %(idx, global_step))
            loss, train_op, y, yp = sess.run([model.loss, model.train_op, model.y, model.scores],
                                          feed_dict={handle: train_handle})

            #print("==1", loss)
            #print("==2", y)
            #print("==3", yp)

            if idx % 10 == 0:
                print("dt: %s, idx: %s, global step: %s, train loss: %s" %(datetime.now(), idx, global_step, loss))
            if global_step % config.period == 0:
                loss_sum = tf.Summary(value=[tf.Summary.Value(
                    tag="model/loss", simple_value=loss), ])
                writer.add_summary(loss_sum, global_step)

            if global_step % config.checkpoint == 0:
                sess.run(tf.assign(model.is_train,
                                   tf.constant(False, dtype=tf.bool)))

                metrics, summ = evaluate_batch(
                    model, dev_total // config.batch_size + 1, dev_eval_file, sess, "dev", handle, dev_handle)
                print(metrics)

                sess.run(tf.assign(model.is_train,
                                   tf.constant(True, dtype=tf.bool)))

                dev_loss = metrics["loss"]
                if dev_loss < loss_save:
                    loss_save = dev_loss
                    patience = 0
                else:
                    patience += 1
                if patience >= config.patience:
                    lr /= 2.0
                    loss_save = dev_loss
                    patience = 0
                sess.run(tf.assign(model.lr, tf.constant(lr, dtype=tf.float32)))
                for s in summ:
                    writer.add_summary(s, global_step)
                writer.flush()

                filename = os.path.join(
                    config.save_dir, "model_{}.ckpt".format(global_step))
                saver.save(sess, filename)
    summary_writer.close()


def evaluate_batch(model, num_batches, eval_file, sess, data_type, handle, str_handle):
    answer_dict = {}
    losses = []
    for _ in range(1, num_batches + 1):
        loss, y, yp = sess.run(
            [model.loss, model.y, model.scores], feed_dict={handle: str_handle})
        losses.append(loss)
    loss = np.mean(losses)

    metrics = {}
    metrics["loss"] = loss
    loss_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/loss".format(data_type), simple_value=metrics["loss"]), ])
    return metrics, [loss_sum,]
