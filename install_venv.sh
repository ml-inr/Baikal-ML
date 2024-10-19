if [ -d "$(pwd)/venv" ]; then
    echo "venv exists"
else
    create_here venv
fi

activate_here venv

# Make sure youre python, tensorflow and cuda versions are compatible!
# The build works for CUDA Version = 11.2 (GPU GeForce GTX 1080 and 2080)
conda install python=3.10 -y
conda install jupyter=7.2 numpy=2.1.2 scipy=1.14 matplotlib=3.9 plotly=2.35 pandas=2.2 polars=1.9.0 joblib=1.4.2 h5py -y
conda install -c conda-forge uproot -y
conda install pytorch=2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia -y
#conda install -c conda-forge tensorflow-gpu=2.11 -y

#pip3 install show_h5