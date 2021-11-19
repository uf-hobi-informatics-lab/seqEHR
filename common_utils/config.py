import enum


class ModelType(enum.Enum):
    M_LSTM = "lstm"
    M_GRU = "gru"
    M_TLSTM = "tlstm"
#     M_TCN = "tcn"
#     M_LSTM_ATT = "lstm-att"
#     M_TLSTM_ATT = "tlstm-att"


class ModelLossMode(enum.Enum):
    BIN = "binary"
    MUL = "multi-classes"


class EmbeddingReductionMode(enum.Enum):
    SUM = "sum"
    AVG = "avg"
    MAX = "max"
    # only when seq is same for all training and test data
    FUSE = "fuse"


class ModelOptimizers(enum.Enum):
    ADAM = "adam"
    SGD = "sgd"


MODEL_TYPE_FLAGS = {
    each.value: each for each in ModelType
}

MODEL_LOSS_MODES = {"bin": ModelLossMode.BIN, "mul": ModelLossMode.MUL}


EMBEDDING_REDUCTION_MODES = {
    "avg": EmbeddingReductionMode.AVG,
    "sum": EmbeddingReductionMode.SUM,
    "max": EmbeddingReductionMode.MAX
}


UNIVERSE_PAD = -1000
