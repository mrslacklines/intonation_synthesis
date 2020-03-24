docker build -f Dockerfile-dev . -t intonation
docker run -it --ipc="private" --shm-size=8000000000 -v /media/tomaszk/DANE/Speech_archive/HTS-demo_AMU_PL_ILO_STRAIGHT:/opt/ml/input/data/training -v /media/tomaszk/DANE/workspace/intonation:/opt/ml/model intonation
