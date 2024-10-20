if [ -d "$(pwd)/venv" ]; then
    echo "venv exists"
else
    create_here venv
fi

activate_here venv

# Make sure youre python, tensorflow and cuda versions are compatible!
# The build works for CUDA Version = 11.2 (GPU GeForce GTX 1080 and 2080)
conda install python=3.10 -y
conda install -c conda-forge pandas polars joblib ipykernel jupyter numpy scipy matplotlib plotly h5py uproot -y
conda install pytorch=2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia -y
#conda install -c conda-forge tensorflow-gpu=2.11 -y

#pip3 install show_h5