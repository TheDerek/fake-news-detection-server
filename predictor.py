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


if __name__ == '__main__':
    p = Predictor()
    print(p.predict('Four major US banks handed almost 1, 000 of their top City staff '
                     'at least €1m (£850, 000) in pay deals last year. Goldman Sachs, the highest profile Wall Street bank, disclosed that 11 of its key staff received at least €5m in 2015. The disclosures by Goldman, JP Morgan, Morgan Stanley and Bank of America Merrill Lynch show that 971 of their staff received €1m in 2015. The information was provided in regulatory disclosures instituted since the 2008 banking crisis, when it became apparent that bankers were being paid huge sums that could not be withheld when banks got into trouble.  Regulations now require banks to spread out bonuses over a number of years. Morgan Stanley, for instance, said that 40% to 60% of its pay deals were deferred over three years, with part of it in shares. The UK arm of Goldman Sachs paid 286 of its staff €1m or more, compared with 262 in 2014. JP Morgan’s disclosures show 301 of its staff received more than €1m, with 11 receiving over €5m. Morgan Stanley’s data shows 198 staff received €1m or more and Bank of America Merrill Lynch shows 186 staff being handed €1m or more.  The disclosures relate to legal entities based in the UK so the majority of the individuals involved will be based in the City, though some may be located in other parts of the EU.   They help to shed light on the pay deals being offered in the City in the wake of the 2008 financial crash and at a time when the sector is facing scrutiny as a result of the vote to leave the EU.  The European Banking Authority (EBA) the   banking regulator, also collates data and in March it announced that London had more than three times as many   bankers as the rest of the EU combined. Overall, the number of high earners across the EU rose 21. 6% to 3, 865 in 2014, up from 3, 178 in 2013.  The EBA’s data covered 2014, the first year of the cap that limits bonuses to 100% of salary, or 200% if shareholders approve. This has had the effect of shifting remuneration towards fixed salaries. In 2014, the average ratio between variable and fixed pay for high earners more than halved to 127% from 317% in 2013.  The EBA will move its headquarters out of London as a result of the vote for Brexit. '))
    pass
