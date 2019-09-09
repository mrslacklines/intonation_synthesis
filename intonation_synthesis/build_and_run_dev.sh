docker build -f Dockerfile-dev . -t intonation-dev
docker run -it -v /media/tomaszk/DANE11/Speech_archive/HTS-demo_AMU_PL_ILO_STRAIGHT:/opt/ml/input/data/training intonation-dev

