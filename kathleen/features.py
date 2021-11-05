import librosa
import numpy as np
import math
import pandas as pd
import json
import re
import os

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

    # Create transcript and features for wav files saved if certain criteria check out
    scotus = []
    batch_flat_transcripts = []
    batch_speakers = []
    batch_start = []
    batch_end = []
    batch_fil = []
    batch_segment = []
    batch_word_count = []
    batch_dur = []
    for docket in saved_dockets:
        transcript, speakers, speaker_roles, times_new = getMeta(docket,data)
        start = [item[0] for item in times_new]
        end = [item[-1] for item in times_new]
        dur = [b - a for a, b in zip(start, end)]
        flat_transcript = [' '.join(item) for item in transcript]
        word_count = [len(item.split()) for item in flat_transcript]
        df = pd.DataFrame({'transcipt':transcript, 'flat_transcript':flat_transcript, 'word_count':word_count,
                           'speakers':speakers, 'speaker_roles':speaker_roles, 'times_new':times_new,
                           'start':start, 'end':end, 'dur':dur})
        scotus_case = df[df['speaker_roles'] == 'scotus_justice']
        scotus_case = scotus_case[scotus_case['word_count'] >= 40]

        scotus_flat_transcript = list(scotus_case['flat_transcript'])
        scotus_speakers = list(scotus_case['speakers'])
        scotus_start = list(scotus_case['start'])
        scotus_end = list(scotus_case['end'])
        scotus_dur = list(scotus_case['dur'])
        scotus_word_count = list(scotus_case['word_count'])
        fil = [docket] * len(scotus_start)
        segment = list(range(len(scotus_start)))
        
        batch_flat_transcripts += scotus_flat_transcript
        batch_speakers += scotus_speakers
        batch_start += scotus_start
        batch_end += scotus_end
        batch_fil += fil
        batch_segment += segment
        batch_word_count += scotus_word_count
        batch_dur += scotus_dur

    start_rounded = [round(item) for item in batch_start]
    end_rounded = [round(item) for item in batch_end]
    return batch_flat_transcripts, batch_speakers, batch_start, batch_end, start_rounded, end_rounded, batch_fil, batch_segment, batch_word_count, batch_dur

def extract_pitch():
    scotus_flat_transcript, scotus_speakers, scotus_start, scotus_end, start_rounded, end_rounded, fil, batch_segment, word_count, dur = get_wav_data()
    f0s = []
    voiced_flags = []
    voiced_probs = []
    onsets_s = []
    onset_means = []
    #var_rofs_1 = []
    #var_rofs_2 = []
    var_rocs = []
    for i in range(len(start_rounded)):
        print(f"Processing {i+1} of {len(start_rounded)} segments")
        current_file = fil[i]
        if (i==0) | ((i>=1) & (current_file != fil[i-1])):
            y, sr = librosa.load(f'{current_file}.wav', sr=librosa.core.get_samplerate(f'{current_file}.wav'))
        i_start = round(scotus_start[i] * sr)
        i_end = round(scotus_end[i] * sr)
        f0, voiced_flag, voiced_prob = librosa.pyin(y[i_start:i_end], fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0s.append(f0)
        voiced_flags.append(voiced_flag)
        voiced_probs.append(voiced_prob)

        # rate of change
        #roc = np.diff(f0)
        #print(rof)
        #var_rof_1 = np.nanvar(rof) # no log
        #var_rof_2 = np.nanvar(np.log(rof)) # log of rate of change
        var_roc = np.nanvar(np.diff(np.log(f0))) # rate of change of log
        #print(var_rof)
        #var_rofs_1.append(var_rof_1)
        #var_rofs_2.append(var_rof_2)
        var_rocs.append(var_roc)

        # onset features
        o_env = librosa.onset.onset_strength(y[i_start:i_end], sr=sr, max_size=5)
        times = librosa.times_like(o_env, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)
        #return (onset_frames.shape[0], o_env.mean())
        onsets = onset_frames.shape[0]
        onset_mean = o_env.mean()

        onsets_s.append(onsets)
        onset_means.append(onset_mean)

    #log_pitch = [np.log(item) for item in f0s]
    #mean_pitch = [np.nanmean(item) for item in f0s]
    log_mean_pitch = [np.nanmean(np.log(item)) for item in f0s]
    #stdev_pitch = [np.nanstd(item) for item in f0s]
    log_stdev_pitch = [np.nanstd(np.log(item)) for item in f0s]

    #data = pd.DataFrame({'transcipt':scotus_flat_transcript, 'speakers':scotus_speakers,
    #                     'start':scotus_start, 'end':scotus_end,'f0':f0s,'voiced_flags':voiced_flags,
    #                     'voiced_probs':voiced_probs,'log_pitch':log_pitch, 'mean_pitch':mean_pitch,
    #                     'log_mean_pitch':log_mean_pitch,'stdev_pitch':stdev_pitch,
    #                     'log_stdev_pitch':log_stdev_pitch})
    data = pd.DataFrame({'transcipt':scotus_flat_transcript, 'speakers':scotus_speakers,'file':fil,'batch':batch_segment,
                         'start':scotus_start, 'end':scotus_end, 'start_rounded':start_rounded, 'end_rounded':end_rounded,
                         'log_mean_pitch':log_mean_pitch,
                         'log_stdev_pitch':log_stdev_pitch, 'rate_of_change_log_variance':var_rocs,
                         'onsets':onsets_s, 'onset_mean':onset_means, 'word_count': word_count, 'duration':dur})

    data.to_csv("pitch.csv")

extract_pitch()
