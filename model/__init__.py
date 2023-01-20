from dataclasses import dataclass
from kospeech.models.deepspeech2.model import DeepSpeech2

@dataclass
class ModelConfig:
    architecture: str = "???"
    teacher_forcing_ratio: float = 1.0
    teacher_forcing_step: float = 0.01
    min_teacher_forcing_ratio: float = 0.9
    dropout: float = 0.3
    bidirectional: bool = False
    joint_ctc_attention: bool = False
    max_len: int = 400


@dataclass
class DeepSpeech2Config(ModelConfig):
    architecture: str = "deepspeech2"
    use_bidirectional: bool = True
    rnn_type: str = "gru"
    hidden_dim: int = 1024
    activation: str = "hardtanh"
    num_encoder_layers: int = 3
