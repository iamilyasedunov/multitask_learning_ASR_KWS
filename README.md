# Multitask learning toolkit

## Install dependencies
### Run script
```
bash start.sh
```
### Or install by commands
```
virtualenv -p $(which python3.8) venv
source venv/bin/activate
python3.8 -m pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
python3.8 -m pip install --ignore-installed ruamel.yaml
python3.8 -m pip install -r requirements.txt
```

## Run training:

### For cmd, kws, asr multitask 
```
source venv_mtl/bin/activate
python3.8 -m run_cmd_kws_asr_multitask_train
```
### For cmd, kws, emo multitask
```
source venv_mtl/bin/activate
python3.8 -m run_cmd_kws_emo_multitask_train
```

### Checkpoints for restore:
```
other/asr_kws_cmd_loss_0.192.pth # asr, kws, cmd
other/kws_cmd_emo_loss_0.385.pth # kws, cmd, emo
```
