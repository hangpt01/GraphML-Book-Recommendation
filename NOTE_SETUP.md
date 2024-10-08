conda create -y -n graphml python=3.8.17
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt

python process_data.py
python gen_dataset.py
