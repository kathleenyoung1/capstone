# TODO: check the word count

import librosa
import numpy as np
import math
import pandas as pd
import json
import re
import os
import soundfile
import time
from joblib import delayed, Parallel, cpu_count
from pydub import AudioSegment

def getMeta(docket, data):

    #get meta data as well as rearrange to desirable formal
    transcript, speakers, speaker_roles, times = data[docket]

    # Flatten times list
    times_new = []
    for t in times:
        flatten = [item for sublist in t for item in sublist]
        times_new.append(flatten)
    # Last element of list is a 0 - cleanup
    del times_new[-1][-1]

    # Flatten speaker_roles list and replace nulls with "Other"
    speaker_roles_clean = []
    for i in speaker_roles:
        if not i:
            speaker_roles_clean.append('Other')
        else:
            speaker_roles_clean.append(i[0])

    # Remove all non-word characters in speakers' names
    speakers =[re.sub(r"[^\w\s]", '', s) for s in speakers]
    # Replace all runs of whitespace with underscorei in speakers' names
    speakers =[re.sub(r"\s+", '_', s) for s in speakers]

    return transcript, speakers, speaker_roles_clean, times_new

def get_wav_data():

    # Get oyez meta data for all dockets out there
    with open(os.getcwd() + '/oyez_metadata.json') as f:
        data = json.load(f)

    # Get names of dockets for which we have wav files for in cwd
    saved_dockets = []
    for file in os.listdir(os.getcwd()):
        if file.endswith(".wav"):
            saved_dockets.append(file.split('.')[0])

    # Create transcript for wav files saved if certain criteria check out
    scotus = []
    batch_flat_transcripts = []
    batch_speakers = []
    batch_start = []
    batch_end = []
    batch_fil = []
    for docket in saved_dockets:
        transcript, speakers, speaker_roles, times_new = getMeta(docket,data)
        start = [item[0] for item in times_new]
        end = [item[-1] for item in times_new]
        flat_transcript = [' '.join(item) for item in transcript]
        word_count = [len(item.split()) for item in flat_transcript]
        df = pd.DataFrame({'transcipt':transcript, 'flat_transcript':flat_transcript, 'word_count':word_count,
                           'speakers':speakers, 'speaker_roles':speaker_roles, 'times_new':times_new,
                           'start':start, 'end':end})
        scotus_case = df[df['speaker_roles'] == 'scotus_justice']
        scotus_case = scotus_case[scotus_case['word_count'] >= 40]

        #scotus.append(scotus_case)
        #scotus_transcript = list(scotus_case['transcript'])
        scotus_flat_transcript = list(scotus_case['flat_transcript'])
        #scotus_word_count = list(scotus_case['word_count'])
        scotus_speakers = list(scotus_case['speakers'])
        #scotus_speaker_roles = list(scotus_case['speaker_roles'])
        #scotus_times_new = list(scotus_case['times_new'])
        scotus_start = list(scotus_case['start'])
        scotus_end = list(scotus_case['end'])
        fil = [docket] * len(scotus_start)

        batch_flat_transcripts += scotus_flat_transcript
        batch_speakers += scotus_speakers
        batch_start += scotus_start
        batch_end += scotus_end
        batch_fil += fil

    #return pd.concat(scotus, ignore_index=True)
    return batch_flat_transcripts, batch_speakers, batch_start, batch_end, batch_fil

def segments_out():
    #scotus_flat_transcript, scotus_speakers, scotus_start, scotus_end, fil = get_wav_data()
    full_data = pd.read_csv('full_data.csv')
    full_data = full_data[(full_data['duration'] <= 20.) & (full_data['file'] != '18-1259.wav')]
    start_idx = list(full_data['start_idx'])
    end_idx = list(full_data['end_idx'])
    fil = list(full_data['file'])
    #start_rounded = [round(item) for item in scotus_start]
    #end_rounded = [round(item) for item in scotus_end]
    #for i in range(len(start_rounded)):
    for i in range(len(start_idx)):
        print(f"Processing {i+1} of {len(start_idx)} segments")
        current_file = fil[i]
        #if (i==0) | ((i>=1) & (current_file != fil[i-1])):
        #    y, sr = librosa.load(f'{current_file}.wav')
        y, sr = librosa.load(f'{current_file}',sr=librosa.core.get_samplerate(f'{current_file}'))
        #i_start = start_rounded[i] * sr
        #i_end = end_rounded[i] * sr
        i_start = start_idx[i]
        i_end = end_idx[i]
        soundfile.write(f'segments5/{current_file[:-4]}_{i}.wav', y[i_start:i_end], sr)

def wav_to_mp3():
    for filename in os.listdir('segments5'):
        if filename.endswith(".wav"):
            AudioSegment.from_wav(f"segments5/{filename}").export(f"mp3s5/{filename[:-4]}.mp3", format="mp3")

segments_out()
wav_to_mp3()
