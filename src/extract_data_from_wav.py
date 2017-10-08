import wave
import random
import time
import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from src.config import *
import h5py
from src.call import *
from src.utilizer import *
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import matplotlib.image as pyimg
import cv2
def extract_call_seg(table_filenames):
    call_segs = []
    for filename in table_filenames:
        subject, session_num = get_info_from_filename(filename)
        fin = open(filename, 'r')
        headline = fin.readline()
        headline = headline.strip().split('\t')
        call_seg = []
        for line in fin.readlines():
            line_split = line.strip().split('\t')
            if len(line_split) > 8:
                call_type = line_split[8]
            else:
                call_type = 'Other'
            call = Call(subject, session_num, int(line_split[2]), float(line_split[3]), float(line_split[4]),call_type)
            call_seg.append(call)
        call_segs.append(call_seg)
    return call_segs

def extract_data(wav_filenames, call_segs, hdf5_filename):
    """
    hdf5 format:
    
    """
    # make hdf5 file
    fout = h5py.File(hdf5_filename, 'w')
    sizeData = [2000, DATA_HEIGHT, DATA_WIDTH, 3, MAX_LSTM_STEP]
    maxShape = (10000, DATA_HEIGHT, DATA_WIDTH, 3, MAX_LSTM_STEP)
    sizeChunk = (BATCH_SIZE, DATA_HEIGHT, DATA_WIDTH, 3, MAX_LSTM_STEP)
    dset_data = fout.create_dataset("dset_data", sizeData, dtype='f', chunks=sizeChunk, maxshape=maxShape)
    sizeLabel = [2000, MAX_LSTM_STEP]
    maxShape = (10000, MAX_LSTM_STEP)
    sizeChunk = (BATCH_SIZE, MAX_LSTM_STEP)
    dset_label = fout.create_dataset("dset_label", sizeLabel, dtype='i', chunks=sizeChunk, maxshape=maxShape)
    sampleCount  = 0
    buf = np.zeros((DATA_HEIGHT, DATA_WIDTH, 3, MAX_LSTM_STEP))
    buf_label = np.zeros(MAX_LSTM_STEP)
    pure_noise_sample = 0
    for filename, call_seg in zip(wav_filenames, call_segs):
        if sampleCount >= 2000:
            break
        fin = wave.open(filename, 'r')
        Fs = fin.getframerate()
        nframes = fin.getnframes()
        nchannels = fin.getnchannels()
        audio_state_query = AudioStateQuery(call_seg, nframes/Fs, WINSIZE_FRAME * SHIFT_FRAME)

        oneSampleFrames = round(((MAX_LSTM_STEP - 1) * SHIFT_FRAME + 1) * WINSIZE_FRAME * Fs)
        for i in range(int(nframes / oneSampleFrames)):
            if sampleCount >= 2000:
                break
            startTime = i * oneSampleFrames / Fs
            stopTime = (i + 1) * oneSampleFrames / Fs
            specLabel = audio_state_query.query(startTime, stopTime, MAX_LSTM_STEP)
            if sum(specLabel) < 3:
                survive = random.uniform(0,1)
                if survive < 0.9:
                    continue
                pure_noise_sample += 1

            sig = fin.readframes(oneSampleFrames)
            sig = np.fromstring(sig, dtype=np.short)
            sig = np.reshape(sig, (-1, nchannels))
            sig = np.transpose(sig)
            
            height = DATA_HEIGHT
            width = round(((MAX_LSTM_STEP -1) * SHIFT_FRAME + 1) * DATA_WIDTH)

            _, _, specPar = spectrogram(
                    sig[:][0], 
                    fs=Fs, window='hann', 
                    nperseg=WINSIZE_SPEC, 
                    noverlap=OVERLAP_SPEC, 
                    scaling='spectrum', 
                    mode='magnitude')
            specPar = np.log(specPar)
            specParPad = cv2.resize(specPar, (width, height))

            _, _, specRef = spectrogram(
                    sig[:][1],
                    fs=Fs, window='hann',
                    nperseg=WINSIZE_SPEC,
                    noverlap=OVERLAP_SPEC,
                    scaling='spectrum',
                    mode='magnitude')
            specRef = np.log(specRef)
            specRefPad = cv2.resize(specRef, (width, height))

            specDifPad = specParPad - specRefPad
            
            for j in range(MAX_LSTM_STEP):
                buf[:,:,0,j] = specParPad[:,j*int(DATA_WIDTH * SHIFT_FRAME) : j*int(DATA_WIDTH * SHIFT_FRAME) + DATA_WIDTH]
                buf[:,:,1,j] = specRefPad[:,j*int(DATA_WIDTH * SHIFT_FRAME) : j*int(DATA_WIDTH * SHIFT_FRAME) + DATA_WIDTH]
                buf[:,:,2,j] = specDifPad[:,j*int(DATA_WIDTH * SHIFT_FRAME) : j*int(DATA_WIDTH * SHIFT_FRAME) + DATA_WIDTH]
                buf_label[j] = specLabel[j]
            print('write sample %g' % sampleCount)
            dset_data[sampleCount,:,:,:,:] = buf
            dset_label[sampleCount,:] = buf_label
            sampleCount += 1
    sizeData[0] = sampleCount
    sizeLabel[0] = sampleCount
    dset_data.resize(sizeData)
    dset_label.resize(sizeLabel)
    fout.close()
    print(pure_noise_sample)

            




if __name__ == '__main__':
    #wav_filenames = ['../data/voc_9606_c_S270.wav']
    #table_filenames = ['../table/SelectionTable_voc_9606_c_S270.txt']
    wav_filenames = ['/mnt/hgfs/VM_share/voc_M93A_c_S187.wav',
                     '/mnt/hgfs/VM_share/voc_M93A_c_S191.wav',
                     '/mnt/hgfs/VM_share/voc_M93A_c_S198.wav']
    table_filenames = ['/mnt/hgfs/VM_share/SelectionTable_voc_M93A_c_S187.txt',
                       '/mnt/hgfs/VM_share/SelectionTable_voc_M93A_c_S191.txt',
                       '/mnt/hgfs/VM_share/SelectionTable_voc_M93A_c_S198.txt']
    #wav_filenames = ['/mnt/hgfs/VM_share/voc_M93A_c_S198.wav']
    #table_filenames = ['/mnt/hgfs/VM_share/SelectionTable_voc_M93A_c_S191.txt']
    hdf5_filename = '../data/train_data_M93A.hdf5'
    call_segs = extract_call_seg(table_filenames)
    extract_data(wav_filenames, call_segs, hdf5_filename)
    

