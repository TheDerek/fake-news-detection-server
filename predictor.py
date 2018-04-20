import os
import tensorflow as tf
import data_helpers
import numpy as np

from tensorflow.contrib import learn

VOCAB_PATH = 'data/vocab'
CHECKPOINT_FILE = 'data/model-55800'
LABELS = {
    1.: 'fake',
    0.: 'real'
}


class Predictor:
    def __init__(self):
        self._vocab_processor = learn.preprocessing.VocabularyProcessor.restore(
            VOCAB_PATH)

    def _transform(self, document: str):
        return np.array(list(self._vocab_processor.transform([document])))

    def predict(self, document):
        x_test = self._transform(document)
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(CHECKPOINT_FILE))
                saver.restore(sess, CHECKPOINT_FILE)

                # Get the placeholders from the graph by name
                input_x = graph.get_operation_by_name("input_x").outputs[0]
                # input_y = graph.get_operation_by_name("input_y").outputs[0]
                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                # Tensors we want to evaluate
                predictions = graph.get_operation_by_name("output/predictions").outputs[0]

                # Generate batches for one epoch
                batches = data_helpers.batch_iter(list(x_test), 64, 1, shuffle=False)

                # Collect the predictions here
                all_predictions = []

                for x_test_batch in batches:
                    batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                    all_predictions = np.concatenate([all_predictions, batch_predictions])

        print(all_predictions)
        return LABELS[all_predictions[0]]
