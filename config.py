import enum


class ModelType(enum.Enum):
    M_LSTM = "clstm"
    M_TLSTM = "ctlstm"


class ModelLossMode(enum.Enum):
    BIN = "binary"
    MUL = "multi-classes"


class ModelOptimizers(enum.Enum):
    ADAM = "adam"
    SGD = "sgd"


MODEL_TYPE_FLAGS = {"clstm": ModelType.M_LSTM, "ctlstm": ModelType.M_TLSTM}
MODEL_LOSS_MODES = {"bin": ModelLossMode.BIN, "mul": ModelLossMode.MUL}