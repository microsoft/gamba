import enum

import sequence_models.constants as constants


class TaskType(enum.Enum):
    LM = "lm"
    OADM = "oadm"


FIM_MIDDLE = "_"
FIM_PREFIX = "("
FIM_SUFFIX = ")"
MSA_ALPHABET_PLUS = constants.MSA_ALPHABET_PLUS + FIM_MIDDLE + FIM_PREFIX + FIM_SUFFIX


# ensure there are no unintentional character overlaps
assert len(MSA_ALPHABET_PLUS) == len(set(MSA_ALPHABET_PLUS))
