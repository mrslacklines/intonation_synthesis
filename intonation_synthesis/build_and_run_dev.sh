docker build -f Dockerfile-dev . -t intonation-dev
docker run -it --ipc="private" --shm-size=8000000000 -p 22:22 -v /media/faqster/DANE/Speech_archive/HTS-demo_AMU_PL_ILO_STRAIGHT:/opt/ml/input/data/training -v /media/faqster/DANE/workspace/intonation:/opt/ml/model -v /media/faqster/DANE/projects/innvestigate/innvestigate:/usr/local/lib/python3.5/dist-packages/innvestigate intonation-dev 

