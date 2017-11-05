import numpy as np
import re
from src.config import *
def get_info_from_filename(filename):
    filename = filename.split('/')[-1]
    divide = re.split('_|\.', filename.strip())
    if divide[0] != 'voc':
        divide = divide[1:]
    prefix = divide[0]
    subject = divide[1]
    if divide[2] == 'c':
        session_num = divide[2] + '_' + divide[3]
    else:
        session_num = divide[2]
    return subject, session_num


class AudioStateQuery():
    def __init__(self, call_seg, duration, res):
        time_resolution = res
        self.res = res
        flag = np.zeros(int(duration / time_resolution))
        for call in call_seg:
            start = int(call.begin_time / time_resolution)
            stop = int(call.end_time / time_resolution)
            print(call.call_type)
            if call.call_type in CALLTYPE_IND_DIC:
                call_type_ind = CALLTYPE_IND_DIC[call.call_type]
            else:
                call_type_ind = len(CALLTYPE_IND_DIC)
            for i in range(start, stop):
                flag[i] = call_type_ind
        self.flag = flag

    def query(self, start_time, stop_time, steps):
        start = int(start_time / self.res)
        stop = int(stop_time / self.res)
        return(self.flag[start : start + steps])

