# clone lda repo
echo 'loading lda-c library' 
git clone https://github.com/blei-lab/lda-c.git

#compiling lda
cd lda-c/
make
cd ..
cp ./lda-c/lda ./app/lda_lib/
rm -rf lda-c
echo 'cleaning up. lda bin copied to ./app/lda_lib/'

echo 'installing deps'
pip install -r requirements.txt    

echo 'starting python application'
python application.py
