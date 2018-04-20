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
        os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)

        self._vocab_processor = learn.preprocessing.VocabularyProcessor.restore(
            VOCAB_PATH)
        self._session = tf.Session(config=session_conf)

        saver = tf.train.import_meta_graph("{}.meta".format(CHECKPOINT_FILE))
        saver.restore(self._session, CHECKPOINT_FILE)

        self._input_x = self._session.graph.get_operation_by_name("input_x").outputs[0]
        self._dropout_keep_prob = self._session.graph.get_operation_by_name(
            "dropout_keep_prob").outputs[0]
        self._predictions = self._session.graph.get_operation_by_name(
            "output/predictions").outputs[0]

    def _transform(self, document: str):
        return np.array(list(self._vocab_processor.transform([document])))

    def predict(self, document):
        x_test = self._transform(document)

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), 64, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = self._session.run(self._predictions,
                {self._input_x: x_test_batch, self._dropout_keep_prob: 1.0})

            all_predictions = np.concatenate([all_predictions, batch_predictions])

        print(all_predictions)
        return LABELS[all_predictions[0]]
