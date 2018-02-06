echo Script to convert local .mp4 file to multiple bitrate .m3u8 for streaming purpose...
echo inputfile: $1
filename=$(basename "$1")
extension="${filename##*.}"
filename="${filename%.*}"
hls_root_dir="/home/hmcheng/nginx/mnt/hls"
directory="$hls_root_dir/$filename"
master_playlist="${directory}.m3u8"
echo creating a directory at $directory
mkdir -p $directory
echo creating the master playlist at $master_playlist
echo "#EXTM3U
#EXT-X-VERSION:3
#EXT-X-STREAM-INF:PROGRAM-ID=1,BANDWIDTH=40000,RESOLUTION=160x90
${filename}/low/index.m3u8
#EXT-X-STREAM-INF:PROGRAM-ID=1,BANDWIDTH=600000,RESOLUTION=640x360
${filename}/src/index.m3u8" > $master_playlist

echo transcoding with ffmpeg...
mkdir ${directory}/src
mkdir ${directory}/low
ffmpeg -i $1 -c:a aac -strict -2 -c:v libx264 -start_number 0 -hls_time 10 -hls_list_size 0 -f hls ${directory}/src/index.m3u8 -c:a aac -strict -2 -c:v libx264 -s 160x90 -start_number 0 -hls_time 10 -hls_list_size 0 -f hls ${directory}/low/index.m3u8

echo Finish! Please see hls client to stream through the address http://ip_of_this_machine/hls/${filename}.m3u8


