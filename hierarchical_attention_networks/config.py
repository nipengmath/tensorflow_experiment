# -*- coding: utf-8 -*-

import os
import tensorflow as tf

from prepro import prepro
from main import train

flags = tf.flags
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

home = os.path.expanduser("./")
train_file = os.path.join(home, "data", "atec", "train.v1.t.json")
dev_file = os.path.join(home, "data", "atec", "train.v1.t.json")
test_file = os.path.join(home, "data", "atec", "train.v1.t.json")
glove_word_file = os.path.join(home, "data", "glove", "glove.token.128.txt")

target_dir = "data"
log_dir = "log/event"
save_dir = "log/model"
answer_dir = "log/answer"

train_record_file = os.path.join(target_dir, "train.tfrecords")
dev_record_file = os.path.join(target_dir, "dev.tfrecords")
test_record_file = os.path.join(target_dir, "test.tfrecords")
word_emb_file = os.path.join(target_dir, "word_emb.json")
char_emb_file = os.path.join(target_dir, "char_emb.json")
train_eval = os.path.join(target_dir, "train_eval.json")
dev_eval = os.path.join(target_dir, "dev_eval.json")
test_eval = os.path.join(target_dir, "test_eval.json")
dev_meta = os.path.join(target_dir, "dev_meta.json")
test_meta = os.path.join(target_dir, "test_meta.json")
word2idx_file = os.path.join(target_dir, "word2idx.json")
char2idx_file = os.path.join(target_dir, "char2idx.json")
answer_file = os.path.join(answer_dir, "answer.json")

if not os.path.exists(target_dir):
    os.makedirs(target_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(answer_dir):
    os.makedirs(answer_dir)

flags.DEFINE_string("mode", "train", "train/debug/test")

flags.DEFINE_string("target_dir", target_dir, "")
flags.DEFINE_string("log_dir", log_dir, "")
flags.DEFINE_string("save_dir", save_dir, "")
flags.DEFINE_string("train_file", train_file, "")
flags.DEFINE_string("dev_file", dev_file, "")
flags.DEFINE_string("test_file", test_file, "")
flags.DEFINE_string("glove_word_file", glove_word_file, "")

flags.DEFINE_string("train_record_file", train_record_file, "")
flags.DEFINE_string("dev_record_file", dev_record_file, "")
flags.DEFINE_string("test_record_file", test_record_file, "")
flags.DEFINE_string("word_emb_file", word_emb_file, "")
flags.DEFINE_string("char_emb_file", char_emb_file, "")
flags.DEFINE_string("train_eval_file", train_eval, "")
flags.DEFINE_string("dev_eval_file", dev_eval, "")
flags.DEFINE_string("test_eval_file", test_eval, "")
flags.DEFINE_string("dev_meta", dev_meta, "")
flags.DEFINE_string("test_meta", test_meta, "")
flags.DEFINE_string("word2idx_file", word2idx_file, "")
flags.DEFINE_string("char2idx_file", char2idx_file, "")
flags.DEFINE_string("answer_file", answer_file, "")

flags.DEFINE_integer("num_classes", 3, "num of classification classes")
## flags.DEFINE_integer("glove_word_size", 7183, "Corpus size for Glove")
flags.DEFINE_integer("glove_dim", 128, "Embedding dimension for Glove")

flags.DEFINE_integer("para_max_num", 5, "max num of para")
flags.DEFINE_integer("para_max_length", 10, "max length of para")

flags.DEFINE_integer("word_count_limit", -1, "Min count for word")
flags.DEFINE_integer("char_count_limit", -1, "Min count for char")

flags.DEFINE_integer("capacity", 15000, "Batch size of dataset shuffle")
flags.DEFINE_integer("num_threads", 4, "Number of threads in input pipeline")

flags.DEFINE_integer("batch_size", 64, "Batch size")

## GRU dimention
flags.DEFINE_integer("hidden_size", 50, "hidden size")

flags.DEFINE_integer("num_steps", 60000, "Number of steps")
flags.DEFINE_integer("checkpoint", 1000, "checkpoint for evaluation")
flags.DEFINE_integer("period", 100, "period to save batch loss")
flags.DEFINE_integer("val_num_batches", 150, "Num of batches for evaluation")
flags.DEFINE_float("init_lr", 0.0002, "Initial lr for Adadelta")
flags.DEFINE_float("clip_gradients", 3.0, "clip_gradients")
flags.DEFINE_float("keep_prob", 0.7, "Keep prob in rnn")
flags.DEFINE_float("grad_clip", 5.0, "Global Norm gradient clipping rate")
flags.DEFINE_integer("patience", 3, "Patience for lr decay")

flags.DEFINE_boolean("is_bucket", False, "Batch size")

tf.app.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.")
tf.app.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.")

# Extensions (Uncomment corresponding line in download.sh to download the required data)


def main(_):
    config = flags.FLAGS
    if config.mode == "train":
        train(config)
    elif config.mode == "prepro":
        prepro(config)
    elif config.mode == "debug":
        config.num_steps = 2
        config.val_num_batches = 1
        config.checkpoint = 1
        config.period = 1
        train(config)
    else:
        print("Unknown mode")
        exit(0)


if __name__ == "__main__":
    tf.app.run()
