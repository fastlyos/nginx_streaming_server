Demo
Feature1: To demonstrate under limited EPC bandwidth, our MEC caching server can increase the video quality significantly
Left monitor (demo_1a.py): View video via internet source (no MEC)
Right monitor (demo_1b.py): View video via our MEC caching server, which proxy the request to the internet video source

Feature2: To demonstrate the bitrate adaptive streaming ability of our HLS video player
Left monitor (demo_2a.py): View video with normal video player. When bandwidth change, the video will lag
Right monitor (demo_2b.py): View video with our hls video player. When bandwidth change, the video will shift to high/low resolution smoothly and adaptively


Installation requirement:
- Python3
- PyQt5
- vlc player (2.2.7)
- mpv player

How to run:
Run "a.py" before "b.py"
for example:
PC1@ubuntu:~$ python3 demo_1a.py
PC2@ubuntu:~$ python3 demo_1b.py