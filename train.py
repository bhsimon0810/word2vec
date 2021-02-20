"""Executable for training Word2vec models.

Example:
    python train.py \
        --filenames=/PATH/TO/FILE/file1.txt,/PATH/TO/FILE/file2.txt \
        --out_dir=/PATH/TO/OUT_DIR/ \
        --batch_size=64 \
        --window_size=5 \

Learned word embeddings will be saved to /PATH/TO/OUT_DIR/embeds.npy
vocabulary saved to /PATH/TO/OUT_DIR/vocab.txt
"""
import os
import tensorflow as tf
import numpy as np

from dataset import Dataset
from model import  Word2Vec

flags = tf.compat.v1.app.flags

flags.DEFINE_integer('epochs', 2, 'Num of epochs to iterate training data.')
flags.DEFINE_integer('batch_size', 512, 'Batch size.')
flags.DEFINE_integer('max_vocab_size', 0, 'Maximum vocabulary size. If > 0, the top `max_vocab_size` most frequent '
                                          'words are kept in vocabulary.')
flags.DEFINE_integer('min_count', 10, 'Words whose counts < `min_count` are not'
                                      ' included in the vocabulary.')
flags.DEFINE_float('sample', 1e-5, 'Subsampling rate.')
flags.DEFINE_integer('window_size', 5, 'Num of words on the left or right side of target word within a window.')

flags.DEFINE_integer('embed_size', 300, 'Length of word vector.')
flags.DEFINE_integer('negatives', 5, 'Num of negative words to sample.')
flags.DEFINE_float('power', 0.75, 'Distortion for negative sampling.')
flags.DEFINE_float('alpha', 0.025, 'Initial learning rate.')
flags.DEFINE_float('min_alpha', 0.001, 'Final learning rate.')
flags.DEFINE_boolean('add_bias', True, 'Whether to add bias term to dotproduct between syn0 and syn1 vectors.')
flags.DEFINE_integer('random_seed', 0, 'Random seed.')

flags.DEFINE_integer('log_per_steps', 10000, 'Every `log_per_steps` steps to output logs.')
flags.DEFINE_list('filenames', '../datasets/pmc.txt,../datasets/pubmed.txt', 'Names of comma-separated input text '
                                                                             'files.')
flags.DEFINE_string('out_dir', '../embeddings', 'Output directory.')

FLAGS = flags.FLAGS


def main(_):
    dataset = Dataset(epochs=FLAGS.epochs,
                      batch_size=FLAGS.batch_size,
                      max_vocab_size=FLAGS.max_vocab_size,
                      min_count=FLAGS.min_count,
                      sample=FLAGS.sample,
                      window_size=FLAGS.window_size)
    dataset.build_vocab(FLAGS.filenames)

    word2vec = Word2Vec(embed_size=FLAGS.embed_size,
                        batch_size=FLAGS.batch_size,
                        negatives=FLAGS.negatives,
                        power=FLAGS.power,
                        alpha=FLAGS.alpha,
                        min_alpha=FLAGS.min_alpha,
                        add_bias=FLAGS.add_bias,
                        random_seed=FLAGS.random_seed)
    to_be_run_dict = word2vec.train(dataset, FLAGS.filenames)

    with tf.compat.v1.Session() as sess:
        sess.run(dataset.iterator_initializer)
        sess.run(tf.compat.v1.tables_initializer())
        sess.run(tf.compat.v1.global_variables_initializer())

        average_loss = 0.
        step = 0
        while True:
            try:
                result_dict = sess.run(to_be_run_dict)
            except tf.errors.OutOfRangeError:
                break

            average_loss += result_dict['loss'].mean()
            if step % FLAGS.log_per_steps == 0:
                if step > 0:
                    average_loss /= FLAGS.log_per_steps
                print('step:', step, 'average_loss:', average_loss,
                      'learning_rate:', result_dict['learning_rate'])
                average_loss = 0.

            step += 1

        embeddings = sess.run(word2vec.embeddings)

    np.save(os.path.join(FLAGS.out_dir, 'embed'), embeddings)
    with open(os.path.join(FLAGS.out_dir, 'vocab.txt'), 'w', encoding="utf-8") as f:
        for w in dataset.table_words:
            f.write(w + '\n')

    print('Word embeddings saved to', os.path.join(FLAGS.out_dir, 'embed.npy'))
    print('Vocabulary saved to', os.path.join(FLAGS.out_dir, 'vocab.txt'))


if __name__ == '__main__':
    tf.compat.v1.app.run()
