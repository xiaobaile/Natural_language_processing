import os
import csv
import tensorflow as tf


flags = tf.flags

# Required parameters
flags.DEFINE_string(
    "data_dir", None, "The input data dir. Should contain the .txv files for the task"
)
flags.DEFINE_string(
    "bert_config_file", None, "The config json file corresponding to the pre-trained BERT model"
)
flags.DEFINE_string(
    "task_name", None, "The name of the task to train"
)
flags.DEFINE_string(
    "output_dir", None, "The output dictionary where the model checkpoints will be written"
)

# Other parameters
flags.DEFINE_string(
    "init_checkpoint", None, "Initial checkpoint (usually from a pre_trained BERT model)"
)
flags.DEFINE_bool(
    "do_lower_case", True, "Whether to lower case the input text. Should be True for uncased"
)

FLAGS = flags.FLAGS


class InputExample(object):
    """ A single training/test example for simple sequence classification"""

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """ A single set of features of data"""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Get a collection of 'InputExample's for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Get a collection of 'InputExample's for the dev set."""
        raise NotImplementedError

    def get_test_examples(self, data_dir):
        """Get a collection of 'InputExample's for prediction."""
        raise NotImplementedError

    def get_label(self):
        """Get the list of labels for this data set."""
        raise NotImplementedError

    @classmethod
    def _read_tsv(cls, input_file, quote_char=None):
        """Read a tab separated value file."""
        with tf.gfile.Open(input_file, mode="r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quote_char)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class XnLiProcessor(DataProcessor):
    """Processor for the XNLI data set."""

    def __init__(self):
        self.language = "zh"

    def get_train_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(
            os.path.join(data_dir, "multinli", "multinli.train.%s.tsv" % self.language)
        )
        examples = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            guid = "train-%d" % i
            text_a =


    def get_dev_examples(self, data_dir):
        # TODO

    def get_test_examples(self, data_dir):
        # TODO

    def get_label(self):
        # TODO










































































































