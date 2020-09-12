import pandas as pd
import tensorflow as tf
from pathlib import Path
import numpy as np


data = pd.read_csv("dataset.csv").values
outfile = Path('/DATA/train_data/')


records_pr_file = 100000
FILES = np.ceil(data.shape[0] / records_pr_file)


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(feature0, feature1, feature2):
  """
  Creates a tf.train.Example message ready to be written to a file.
  """
  # Create a dictionary mapping the feature name to the tf.train.Example-compatible
  # data type.
  feature = {
      'sequence': _bytes_feature(feature0.encode('utf-8')),
      'timing': _bytes_feature(feature1.encode('utf-8')),
      'target': _bytes_feature(feature2.encode('utf-8')),
  }

  # Create a Features message using tf.train.Example.

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()

# Write the `tf.train.Example` observations to the files.
for i in range(FILES):
    print("Writing file train{}.tfrecord".format(i))
    with tf.io.TFRecordWriter('train{}.tfrecord'.format(i)) as writer:
        for j in range(records_pr_file):
            try:
                example = serialize_example(*data[j + i*records_pr_file])
                writer.write(example)
            except(IndexError):
                break
