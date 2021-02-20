import numpy as np
import tensorflow as tf


class Word2Vec(object):
    """
    Word2Vec Model.
    """

    def __init__(self, embed_size, batch_size, negatives, power,
                 alpha, min_alpha, add_bias, random_seed):
        """Constructor.

        Args:
            embed_size: int scalar, embedding dimension.
            batch_size: int scalar.
            negatives: int scalar, number of negative words to sample.
            power: float scalar, distortion on word frequency when sampling negative words.
            alpha: float scalar, initial learning rate.
            min_alpha: float scalar, final learning rate.
            add_bias: bool scalar, whether to add bias term to dot-product between syn0 and syn1 vectors.
            random_seed: int scalar, random_seed.
        """
        self._embed_size = embed_size
        self._batch_size = batch_size
        self._negatives = negatives
        self._power = power
        self._alpha = alpha
        self._min_alpha = min_alpha
        self._add_bias = add_bias
        self._random_seed = random_seed

        self._syn0 = None

    @property
    def embeddings(self):
        return self._syn0

    def _build_loss(self, inputs, labels, unigram_counts, scope=None):
        """Build the computation graph.

        Args:
            inputs: int tensor, with shape [batch_size, ].
            labels: int tensor, with shape [batch_size, ].
            unigram_counts: list of int, holding word frequencies.

        Returns:
            loss: float tensor, with shape [batch_size, ].
        """
        syn0, syn1, biases = self._create_embeddings(len(unigram_counts))
        self._syn0 = syn0
        with tf.compat.v1.variable_scope(scope, 'Loss'):
            loss = self._negative_sampling_loss(unigram_counts, inputs, labels, syn0, syn1, biases)
            return loss

    def train(self, dataset, filenames):
        """Add `train_op` to the computation graph.

        Args:
            dataset: `Dataset` instance defined in `dataset.py`
            filenames: list of strings, holding file names.

        Returns:
            to_be_run_dict: `train_op`, `loss` and `learning_rate` to be run.
        """
        tensor_dict = dataset.get_tensor_dict(filenames)
        inputs, labels = tensor_dict['inputs'], tensor_dict['labels']
        global_step = tf.compat.v1.train.get_or_create_global_step()
        learning_rate = tf.maximum(self._alpha * (1 - tensor_dict['progress'][0]) +
                                   self._min_alpha * tensor_dict['progress'][0], self._min_alpha)

        loss = self._build_loss(inputs, labels, dataset.unigram_counts)
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step) 

        to_be_run_dict = {'train_op': train_op,
                          'loss': loss,
                          'learning_rate': learning_rate}
        return to_be_run_dict

    def _create_embeddings(self, vocab_size, scope=None):
        """Create word2vec embeddings.

        Args:
            vocab_size: vocabulary size.

        Returns:
            syn0: float tensor of shape [vocab_size, embed_size], input word embeddings (i.e. weights of hidden layer).
            syn1: float tensor of shape [vocab_size, embed_size], output word embeddings (i.e. weights of output layer).
            biases: float tensor of shape [vocab_size], biases added onto the logits.
        """
        with tf.compat.v1.variable_scope(scope, 'Embedding'):
            syn0 = tf.compat.v1.get_variable('syn0',
                                             initializer=tf.random.uniform([vocab_size, self._embed_size],
                                                                           -0.5 / self._embed_size,
                                                                           0.5 / self._embed_size,
                                                                           seed=self._random_seed))
            syn1 = tf.compat.v1.get_variable('syn1',
                                             initializer=tf.random_uniform([vocab_size, self._embed_size], -0.1, 0.1))
            biases = tf.compat.v1.get_variable('biases', initializer=tf.zeros([vocab_size]))
        return syn0, syn1, biases

    def _negative_sampling_loss(self, unigram_counts, inputs, labels, syn0, syn1, biases):
        """Build negative sampling loss.

        Args:
            unigram_counts: list of int, holding the word frequencies.
            inputs: int tensor of shape [batch_size, ].
            labels: int tensor of [batch_size, ].
            syn0: float tensor of shape [vocab_size, embed_size], input word embeddings (i.e. weights of hidden layer).
            syn1: float tensor of shape [vocab_size, embed_size], output word embeddings (i.e. weights of output layer).
            biases: float tensor of shape [vocab_size], biases added onto the logits.

        Returns:
            loss: float tensor of shape [batch_size, sample_size + 1].
        """
        sampled_values = tf.nn.fixed_unigram_candidate_sampler(
            true_classes=tf.expand_dims(labels, axis=1),
            num_true=1,
            num_sampled=self._batch_size * self._negatives,
            unique=True,
            range_max=len(unigram_counts),
            distortion=self._power,
            unigrams=unigram_counts)

        sampled = sampled_values.sampled_candidates
        sampled_mat = tf.reshape(sampled, [self._batch_size, self._negatives])
        inputs_syn0 = tf.gather(syn0, inputs)  # [batch_size, embed_size]
        true_syn1 = tf.gather(syn1, labels) # [batch_size, embed_size]
        sampled_syn1 = tf.gather(syn1, sampled_mat) # [batch_size, k, embed_size]

        true_logits = tf.reduce_sum(tf.multiply(inputs_syn0, true_syn1), 1)  # [batch_size]
        sampled_logits = tf.reduce_sum(tf.multiply(tf.expand_dims(inputs_syn0, 1), sampled_syn1), 2)  # [batch_size, k]

        if self._add_bias:
            true_logits += tf.gather(biases, labels)  # [batch_size]
            sampled_logits += tf.gather(biases, sampled_mat)  # [batch_size, k]

        # cross entropy loss
        true_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(true_logits),
                                                                     logits=true_logits)
        sampled_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(sampled_logits),
                                                                        logits=sampled_logits)
        loss = tf.concat([tf.expand_dims(true_cross_entropy, 1), sampled_cross_entropy], 1)

        return loss
