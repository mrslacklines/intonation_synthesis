git clone https://github.com/CSTR-Edinburgh/merlin.git && mv merlin merlin_repo && ln -s merlin_repo/src/ merlin && cd merlin_repo && pip install -r requirements.txt 
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && tar xvf ta-lib-0.4.0-src.tar.gz && cd ta-lib && ./configure && make && sudo make install
python train.py
