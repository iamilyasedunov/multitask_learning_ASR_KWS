virtualenv -p $(which python3.8) venv_mtl
source venv_mtl/bin/activate
python3.8 -m pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
python3.8 -m pip install --ignore-installed ruamel.yaml
python3.8 -m pip install -r requirements.txt
gdown --id 1KmWWWsexI1n_7E-pcn0-UalrsGKQjcJt -O other/asr_kws_cmd_loss_0.192.pth
gdown --id 1MFMJLKsk1M_OBD4OA5jlfZqZPG3zRW4M -O other/kws_cmd_emo_loss_0.385.pth
