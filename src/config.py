WINSIZE_SPEC = 512 # number of points used in STFT, 256 points in frequency domain
OVERLAP_SPEC = 462 # shift 50 points
WINSIZE_FRAME = 0.05 # sec, 50 points in spectrum if FS = 50000 and SHIFT_SPEC = 50
SHIFT_FRAME = 0.8 # percent
MAX_LSTM_STEP = 100 # 4 secs: 160 * 0.05 * 0.8
DATA_HEIGHT = 64
DATA_WIDTH = 64
BATCH_SIZE = 128

FS = 50000

CALLTYPE_IND_DIC = {'Noise': 0, 'Trill': 1, 'Phee': 2, 'Trillphee': 3, 'Twitter':4, 'Sd-peep': 5, 'Others': 6}

