if [ -d "$(pwd)/venv" ]; then
    echo "venv exists"
else
    create_here venv
fi

activate_here venv

# Make sure youre python, tensorflow and cuda versions are compatible!
# The build works for CUDA Version = 11.2 (GPU GeForce GTX 1080 and 2080)
conda install python=3.10 -y
conda install jupyter numpy scipy matplotlib plotly pandas h5py sqlalchemy tqdm -y
conda install -c conda-forge uproot -y
#conda install -c conda-forge tensorflow-gpu=2.11 -y

pip3 install show_h5