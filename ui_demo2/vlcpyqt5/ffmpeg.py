import ffmpeg
stream = ffmpeg.input('/home/hmcheng/small.mp4')
ffmpeg.run(stream)