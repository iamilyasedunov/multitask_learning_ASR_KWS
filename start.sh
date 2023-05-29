virtualenv -p $(which python3.8) venv_mtl
source venv_mtl/bin/activate
python3.8 -m pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
python3.8 -m pip install --ignore-installed ruamel.yaml
python3.8 -m pip install -r requirements.txt
