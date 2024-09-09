from MPDLinear.model.MPDLinear_v1 import MPDLinear_v1
from MPDLinear.model.MPDLinear_v10 import MPDLinear_v10
from MPDLinear.model.MPDLinear_v11 import MPDLinear_v11
from MPDLinear.model.MPDLinear_v12 import MPDLinear_v12
from MPDLinear.model.MPDLinear_v13 import MPDLinear_v13
from MPDLinear.model.MPDLinear_v14 import MPDLinear_v14
from MPDLinear.model.MPDLinear_v2 import MPDLinear_v2
from MPDLinear.model.MPDLinear_v3 import MPDLinear_v3
from MPDLinear.model.DLinear import DLinear
from MPDLinear.model.MPDLinear_SOTA import MPDLinear_SOTA
from MPDLinear.model.MPDLinear_v4 import MPDLinear_v4
from MPDLinear.model.MPDLinear_v5 import MPDLinear_v5
from MPDLinear.model.MPDLinear_v6 import MPDLinear_v6
from MPDLinear.model.MPDLinear_v7 import MPDLinear_v7
from MPDLinear.model.MPDLinear_v8 import MPDLinear_v8
from MPDLinear.model.MPDLinear_v9 import MPDLinear_v9


class ModelDict(object):
    def __init__(self):
        self.model_dict = {
            'DLinear': DLinear,
            'MPDLinear_v1': MPDLinear_v1,
            'MPDLinear_v2': MPDLinear_v2,
            'MPDLinear_v3': MPDLinear_v3,
            'MPDLinear_v4': MPDLinear_v4,
            'MPDLinear_v5': MPDLinear_v5,
            'MPDLinear_v6': MPDLinear_v6,
            'MPDLinear_v7': MPDLinear_v7,
            'MPDLinear_v8': MPDLinear_v8,
            'MPDLinear_v9': MPDLinear_v9,
            'MPDLinear_v10': MPDLinear_v10,
            'MPDLinear_v11': MPDLinear_v11,
            'MPDLinear_v12': MPDLinear_v12,
            'MPDLinear_v13': MPDLinear_v13,
            'MPDLinear_v14': MPDLinear_v14,
            'MPDLinear_SOTA': MPDLinear_SOTA
        }

    def get_model_dict(self):
        return self.model_dict