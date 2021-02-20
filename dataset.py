import pickle
import itertools
import collections
import numpy as np
import tensorflow as tf
from functools import partial

# from utils import read_lines_from_file_as_data_chunks

OOV_ID = -1
# CHUNK_SIZE = 100000


class Dataset(object):
    """
    Dataset for generating tensors holding word indices to train `Word2Vec` models.
    """

    def __init__(self,
                 epochs=5,
                 batch_size=128,
                 max_vocab_size=0,
                 min_count=10,
                 sample=1e-3,
                 window_size=5):
        """Constructor.

        Args:
            epochs: int scalar.
            batch_size: int scalar.
            max_vocab_size: int scalar, maximum vocabulary size. If > 0, the top `max_vocab_size` most frequent
                            words are kept in vocabulary.
            min_count: int scalar, words with frequency less than `min_count` are not included in the vocabulary.
            sample: float scalar, subsampling rate.
            window_size: int scalar, number of words on the left or right side within a sliding window.
        """
        self._epochs = epochs
        self._batch_size = batch_size
        self._max_vocab_size = max_vocab_size
        self._min_count = min_count
        self._sample = sample
        self._window_size = window_size

        self._iterator_initializer = None
        self._table_words = None
        self._unigram_counts = None
        self._keep_probs = None
        self._corpus_size = None

    @property
    def iterator_initializer(self):
        return self._iterator_initializer

    @property
    def table_words(self):
        return self._table_words

    @property
    def unigram_counts(self):
        return self._unigram_counts

    def _build_raw_vocab(self, filenames):
        """Build raw vocabulary from text files.

        Args:
            filenames: list of strings, holding file names.

        Returns:
            raw_vocab: list of (word, frequency) tuples in descending order of word frequency.
        """
        map_open = partial(open, encoding="utf-8")
        lines = itertools.chain(*map(map_open, filenames))
        raw_vocab = collections.Counter()
        for line in lines:
            raw_vocab.update(line.strip().split())
        raw_vocab = raw_vocab.most_common()
        if self._max_vocab_size > 0:
            raw_vocab = raw_vocab[:self._max_vocab_size]
        return raw_vocab

    def build_vocab(self, filenames):
        """Build vocabulary.
        After building the raw vocabulary, the following attributes also be constructed.
            - table_words: list of string, holding the vocabulary words.
            - unigram_counts: list of int, holding the word frequencies.
            - keep_probs: list of float, holding word's prob for subsampling.

        Args:
            filenames: list of strings, containing file names.
        """
        raw_vocab = self._build_raw_vocab(filenames)
        raw_vocab = [(w, c) for w, c in raw_vocab if c >= self._min_count]
        self._corpus_size = sum(list(zip(*raw_vocab))[1])

        self._table_words = []
        self._unigram_counts = []
        self._keep_probs = []
        for word, count in raw_vocab:
            frac = count / float(self._corpus_size)
            keep_prob = (np.sqrt(frac / self._sample) + 1) * (self._sample / frac)
            keep_prob = np.minimum(keep_prob, 1.0)
            self._table_words.append(word)
            self._unigram_counts.append(count)
            self._keep_probs.append(keep_prob)

    def _prepare_inputs_labels(self, tensor):
        """Split `tensor` to `inputs` and `labels`

        Args:
            tensor: returned by `generate_instances`, holding indices of both target word and it's context word.

        Returns:
             inputs: tensor, holding indices of target word
             labels: tensor, holding indices of context word
        """
        tensor.set_shape([self._batch_size, 2])
        inputs, labels = tensor[:, :1], tensor[:, 1:]
        return inputs, labels

    def get_tensor_dict(self, filenames):
        """Generates tensor dict for training.

        Args:
            filenames: list of strings, containing file names.

        Returns:
            a tensor dict for training the `Word2Vec` model.
        """
        if not self._table_words or not self._unigram_counts or not self._keep_probs:
            raise ValueError('`table_words`, `unigram_counts`, and `keep_probs` must',
                             'be set by calling `build_vocab()`')

        table_words = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(self._table_words), default_value=OOV_ID)
        keep_probs = tf.constant(self._keep_probs)

        '''
        # if the size of training data is too large, use this block of code
        # to load the data.
        num_sents = 0

        def count_lines(data, eof, file_name):
            """Count the num lines in the text file. Out-of-memory error will be raised if directly open.

            Args:
                data: a single line of the text file.
                eof: end of chunk size.
                filename: text file to be counted.
            """
            nonlocal num_sents
            # check if end of file reached
            if not eof:
                # process data, data is one single line of the file
                num_sents += 1

        for fn in filenames:
            read_lines_from_file_as_data_chunks(fn, chunk_size=CHUNK_SIZE, callback=count_lines)
        '''

        num_sents = sum([len(list(open(fn, encoding="utf-8"))) for fn in filenames])
        num_sents = self._epochs * num_sents

        # include epoch number, like progress
        a_zip = tf.data.TextLineDataset(filenames).repeat(self._epochs)
        b_zip = tf.range(1, 1 + num_sents) / num_sents
        c_zip = tf.repeat(tf.range(1, 1 + self._epochs), int(num_sents / self._epochs))

        dataset = tf.data.Dataset.zip((a_zip,
                                       tf.data.Dataset.from_tensor_slices(b_zip),
                                       tf.data.Dataset.from_tensor_slices(c_zip)))

        # transform raw words into id sequences
        dataset = dataset.map(lambda sent, progress, epoch:
                              (get_word_indices(sent, table_words), progress, epoch))
        # subsample sentences based on word frequency
        dataset = dataset.map(lambda indices, progress, epoch:
                              (subsample(indices, keep_probs), progress, epoch))
        # filter sentences with length less than 3
        dataset = dataset.filter(lambda indices, progress, epoch:
                                 tf.greater(tf.size(indices), 3))
        # transform id sequences into [current_word_id, left_or_right_word_id]
        dataset = dataset.map(lambda indices, progress, epoch: (
            generate_instances(
                indices, self._window_size), progress, epoch))

        # update the progress and epoch
        dataset = dataset.map(lambda instances, progress, epoch: (
            instances, tf.fill(tf.shape(instances)[:1], progress),
            tf.fill(tf.shape(instances)[:1], epoch)))

        dataset = dataset.flat_map(lambda instances, progress, epoch:
                                   tf.data.Dataset.from_tensor_slices((instances, progress, epoch)))
        dataset = dataset.batch(self._batch_size, drop_remainder=True)

        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        self._iterator_initializer = iterator.initializer
        tensor, progress, epoch = iterator.get_next()
        progress.set_shape([self._batch_size])
        epoch.set_shape([self._batch_size])

        inputs, labels = self._prepare_inputs_labels(tensor)
        inputs = tf.squeeze(inputs, axis=1)
        labels = tf.squeeze(labels, axis=1)
        return {'inputs': inputs, 'labels': labels, 'progress': progress, 'epoch': epoch}


def get_word_indices(sent, table_words):
    """Transform a sentence into a list of word indices.

    Args:
        sent: a string sentence where words are space-delimited.
        table_words: a lookup tabel for transforming words to indices.

    Returns:
        indices: transformed word indices.
    """
    words = tf.string_split([sent]).values
    indices = tf.cast(table_words.lookup(words), dtype=tf.int32)
    return indices


def subsample(indices, keep_probs):
    """Filters OOV words and subsamples on words in a sentence. Words with high frequencies have lower keep probs.

    Args:
        indices: transformed word indices.
        keep_probs: keep prob when randomly droping words.

    Returns:
        indices: subsampled word indices.
    """
    indices = tf.boolean_mask(indices, tf.not_equal(indices, OOV_ID))
    keep_probs = tf.gather(keep_probs, indices)
    randvars = tf.random.uniform(tf.shape(keep_probs), 0, 1)
    indices = tf.boolean_mask(indices, tf.less(randvars, keep_probs))
    return indices


def generate_instances(indices, window_size):
    """Generating instances to be passed to the `Word2Vec` model.

    Args:
        indices: subsampled word indices.
        window_size: sliding window size.

    Returns:
        instances: each instance with shape [2], holding indices of both target word and context word.
    """

    def per_target_fn(index, init_array):
        reduced_size = tf.random.uniform([], maxval=window_size, dtype=tf.int32)
        left = tf.range(tf.maximum(index - window_size + reduced_size, 0), index)
        right = tf.range(index + 1,
                         tf.minimum(index + 1 + window_size - reduced_size, tf.size(indices)))
        context = tf.concat([left, right], axis=0)
        context = tf.gather(indices, context)

        # construct training instances within a sliding window.
        window = tf.stack([tf.fill(tf.shape(context), indices[index]), context], axis=1)

        return index + 1, init_array.write(index, window)

    size = tf.size(indices)
    init_array = tf.TensorArray(tf.int32, size=size, infer_shape=False)
    _, result_array = tf.while_loop(lambda i, ta: i < size,
                                    per_target_fn,
                                    [0, init_array],
                                    back_prop=False)
    instances = tf.cast(result_array.concat(), tf.int64)
    return instances
