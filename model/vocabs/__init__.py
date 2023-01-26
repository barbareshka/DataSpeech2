class Vocabulary(object):
    """
    Note:
        Do not use this class directly, use one of the sub classes.
    """
    def __init__(self, *args, **kwargs):
        self.sos_id = None
        self.eos_id = None
        self.pad_id = None
        self.blank_id = None

    def label_to_string(self, labels):
        raise NotImplementedError


from kospeech.vocabs.ksponspeech import KsponSpeechVocabulary
from kospeech.vocabs.librispeech import LibriSpeechVocabulary
