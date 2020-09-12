#%%
import tensorflow as tf
import numpy as np
from model import Transformer, CustomSchedule
from pathlib import Path
import pandas as pd

DATAPATH =Path("DATA/")
# train = pd.read_csv(DATAPATH.joinpath("train.csv"))
# val = pd.read_csv(DATAPATH.joinpath("train.csv"))
products = pd.read_csv(DATAPATH.joinpath("products/products.csv"))
dataset_path = Path('dataset.csv')

#%%

MAXLEN = 200
NUM_LAYERS = 6
DIM = 128
HEADS = 8
BATCH_SIZE = 128

EOS_ID = products.product_id.max() + 1
SOS_ID  = EOS_ID + 1
VOCAB_SIZE = SOS_ID + 1
#%%
def process_input(x1, x2, y):

    # Start/end token
    x1 = str(SOS_ID) + '_' + x1 + '_' + str(EOS_ID)
    y = str(SOS_ID) + '_' + y + '_' + str(EOS_ID)

    # Turn to integer vector
    x1 = tf.strings.to_number(tf.strings.split(x1, sep='_'), out_type = tf.int64)
    x2 = tf.strings.to_number(tf.strings.split(x2, sep='_'), out_type = tf.int64)
    y = tf.strings.to_number(tf.strings.split(y, sep='_'), out_type = tf.int64)

    return x1, x2, y

def tf_process_inputs(x1,x2,y):
    x1_, x2_, y_ = tf.py_function(process_input, [x1,x2,y], [tf.int64, tf.int64, tf.int64])
    x1_.set_shape([None])
    x2_.set_shape([None])
    y_.set_shape([None])
    return x1_, x2_, y_

def _parse_function(example_proto):
  # Parse the input `tf.train.Example` proto using the dictionary above.

    feature_description = {
    'sequence': tf.io.FixedLenFeature([], tf.string),
    'timing': tf.io.FixedLenFeature([],tf.string),
    'target': tf.io.FixedLenFeature([], tf.string)
    }
    return tf.io.parse_single_example(example_proto, feature_description)

filenames = ["train{}.tfrecord".format(i) for i in range(31)]
raw_dataset = tf.data.TFRecordDataset(filenames)

parsed_data = raw_dataset.map(_parse_function)
parsed_data = parsed_data.map(lambda x: tf_process_inputs(x['sequence'], x['timing'], x['target']))
parsed_data = parsed_data.shuffle(100000).padded_batch(BATCH_SIZE)
parsed_data = parsed_data.prefetch(tf.data.experimental.AUTOTUNE)

#%%
parsed_data.take(1)
#%%



# Model definition
transformer = Transformer(num_layers = NUM_LAYERS, dim = DIM, heads = HEADS, vocab_size = VOCAB_SIZE)

# Define optimizer
learning_rate = CustomSchedule(DIM)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)

# Loss/metric tracking
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none') # reduction = 'none' in order to mask out padding tokens

#%%
vocab_size = 1000
vocab = np.arange(vocab_size)

x = np.random.choice(vocab, 10*5).reshape((10,5))
y = np.random.choice(vocab, 10*5).reshape((10,5))

transformer = Transformer(num_layers=8, dim = 100, heads = 5, vocab_size=vocab_size)
out = transformer(x, y)
#%%
tf.reduce_sum(loss_object(y, out))
#%%
transformer.summary() 
#%%


def loss_function(real, pred):
    # Mask out padding tokens from loss function
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

# Checkpoints
checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')



def train_step(x, y):
    # Get the right start/end tokens.
    y_in = y[:,:-1]
    y_out = y[:,1:]

    with tf.GradientTape() as tape:
        preds = transformer(x, y_in, training = True)
        loss = loss_function(y_out, preds)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    
    train_loss(loss)
    train_accuracy(tar_real, predictions)

