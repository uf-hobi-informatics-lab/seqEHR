import enum


class ModelType(enum.Enum):
    M_LSTM = "lstm"
    M_GRU = "gru"
    M_TLSTM = "tlstm"


class ModelLossMode(enum.Enum):
    BIN = "binary"
    MUL = "multi-classes"


class EmbeddingReductionMode(enum.Enum):
    SUM = "sum"
    FUSE = "fuse"


class ModelOptimizers(enum.Enum):
    ADAM = "adam"
    SGD = "sgd"


MODEL_TYPE_FLAGS = {
    "lstm": ModelType.M_LSTM,
    "tlstm": ModelType.M_TLSTM,
    "gru": ModelType.M_GRU,
}
MODEL_LOSS_MODES = {"bin": ModelLossMode.BIN, "mul": ModelLossMode.MUL}