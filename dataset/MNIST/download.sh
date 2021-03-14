mkdir -p raw
mkdir -p processed

cd raw

wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

7z x train-images-idx3-ubyte.gz
7z x train-labels-idx1-ubyte.gz
7z x t10k-images-idx3-ubyte.gz
7z x t10k-labels-idx1-ubyte.gz

cd ..

python process.py