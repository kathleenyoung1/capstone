# using these scripts

1. cd to your scratch folder and install conda (make sure you're in scratch! librosa is too big to install in home):
- wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
- bash Miniconda3-latest-Linux-x86_64.sh
- Make sure the location for miniconda3 is in your scratch folder
- export PATH="/scratch/netid/miniconda3:$PATH"

2. Create a capstone environment & install librosa
- conda create --name capstone
- conda install librosa

3. Follow steps 1-3 from Jeff's SCOTUS folder to get all of the wav files and metadata in your scratch: https://github.com/JeffT13/rd-diarization/tree/main/SCOTUS

4. For get_pitch.py:
-python get_pitch.py
-This will create a CSV with all of the pitch DataFrame

For create_segments.py:
-Create a directory called "segments" in the directory that contains create_segments.py
-python create_segments.py

NOTE: I made a lot of mistakes setting this up so it's possible I missed a necessary step in these instructions! Please let me know if anything goes wrong and I can add it in.
