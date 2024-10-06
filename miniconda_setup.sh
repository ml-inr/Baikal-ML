mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
echo -e '\nconda deactivate\n' >> ~/.bashrc
echo -e '\ncreate_here(){\n\tconda create -p $(pwd)/$1 python=3.10\n}' >> ~/.bashrc
echo -e '\nactivate_here(){\n\tconda activate $(pwd)/$1\n}' >> ~/.bashrc
exec bash