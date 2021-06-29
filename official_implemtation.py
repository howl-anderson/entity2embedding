import io
import re
import string

import tensorflow as tf
import tqdm

num_ns = 4
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE


# Generates skip-gram pairs with negative sampling for a list of sequences
# (int-encoded sentences) based on window size, number of negative samples
# and vocabulary size.
def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
    # Build the sampling table for vocab_size tokens.
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

    # Iterate over all sequences (sentences) in dataset.
    for sequence in tqdm.tqdm(sequences):

        # Generate positive skip-gram pairs for a sequence (sentence).
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            sequence,
            vocabulary_size=vocab_size,
            sampling_table=sampling_table,
            window_size=window_size,
            negative_samples=0,
        )

        # Iterate over each positive skip-gram pair to produce training examples
        # with positive context word and negative samples.
        for target_word, context_word in positive_skip_grams:
            context_class = tf.expand_dims(
                tf.constant([context_word], dtype="int64"), 1
            )
            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=context_class,
                num_true=1,
                num_sampled=num_ns,
                unique=True,
                range_max=vocab_size,
                seed=SEED,
                name="negative_sampling",
            )

            # Build context and label vectors (for one target word)
            negative_sampling_candidates = tf.expand_dims(
                negative_sampling_candidates, 1
            )

            context = tf.concat([context_class, negative_sampling_candidates], 0)
            label = tf.constant([1] + [0] * num_ns, dtype="int64")
            target = tf.constant([target_word], dtype="int64")

            # import pdb; pdb.set_trace()
            # return target_word
            yield (target, context), label


path_to_file = tf.keras.utils.get_file(
    "shakespeare.txt",
    "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt",
)
# path_to_file = "./text8"

text_ds = tf.data.TextLineDataset(path_to_file)
# filter out empty line
text_ds = text_ds.filter(lambda x: tf.cast(tf.strings.length(x), bool))

# Now, create a custom standardization function to lowercase the text and
# remove punctuation.
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    return tf.strings.regex_replace(
        lowercase, "[%s]" % re.escape(string.punctuation), ""
    )


# Define the vocabulary size and number of words in a sequence.
vocab_size = 4096
sequence_length = None

# Use the text vectorization layer to normalize, split, and map strings to
# integers. Set output_sequence_length length to pad all samples to same length.
vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length,
)

vectorize_layer.adapt(text_ds.batch(1024))
# Save the created vocabulary for reference.
inverse_vocab = vectorize_layer.get_vocabulary()
print(inverse_vocab[:20])

# Vectorize the data in text_ds.
text_vector_ds = text_ds.batch(1024).prefetch(AUTOTUNE).map(vectorize_layer).unbatch()

sequences = text_vector_ds.as_numpy_iterator()

traning_data_generator = generate_training_data(
    sequences=sequences, window_size=2, num_ns=4, vocab_size=vocab_size, seed=SEED
)

BATCH_SIZE = 1024
BUFFER_SIZE = 10000
dataset = tf.data.Dataset.from_generator(
    lambda: traning_data_generator,
    output_signature=(
        (
            tf.TensorSpec(shape=(1,), dtype=tf.int64),
            tf.TensorSpec(shape=(5, 1), dtype=tf.int64),
        ),
        tf.TensorSpec(shape=(5,), dtype=tf.int64),
    ),
)
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
print(dataset)


class Word2Vec(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.target_embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim, input_length=1, name="w2v_embedding"
        )
        self.context_embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim, input_length=num_ns + 1
        )
        self.dots = tf.keras.layers.Dot(axes=(3, 2))
        self.flatten = tf.keras.layers.Flatten()

    def call(self, pair):
        target, context = pair
        word_emb = self.target_embedding(target)
        context_emb = self.context_embedding(context)
        dots = self.dots([context_emb, word_emb])
        logits = self.flatten(dots)
        # import pdb; pdb.set_trace()
        return logits


embedding_dim = 128
word2vec = Word2Vec(vocab_size, embedding_dim)
word2vec.compile(
    optimizer="adam",
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")

word2vec.fit(dataset, epochs=20, callbacks=[tensorboard_callback])
