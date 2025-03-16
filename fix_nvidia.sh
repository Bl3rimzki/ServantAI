
# 8️⃣ Python 3.10 mit pyenv
sudo apt-get install -y \
    make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev curl \
    llvm libncursesw5-dev xz-utils tk-dev libxml2-dev \
    libxmlsec1-dev libffi-dev liblzma-dev python3-venv

curl https://pyenv.run | bash
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
. ~/.bashrc

pyenv install 3.10.14
pyenv virtualenv 3.10.14 servantai-env
pyenv activate servantai-env


# 🔟 Requirements installieren
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 1️⃣1️⃣ FastSAM & BoxMOT holen
if [ ! -d "FastSAM" ]; then
    git clone https://github.com/CASIA-IVA-Lab/FastSAM.git
    cd FastSAM
    pip install -e .
fi

cd ~/Git/ServantAI
if [ ! -d "BoxMOT" ]; then
    git clone https://github.com/bochinski/BoxMOT.git
    cd BoxMOT
    pip install -e .
fi

# 1️⃣2️⃣ ServantAI Testlauf
cd ~/Git/ServantAI
echo "🚀 Starte ServantAI Testlauf:"
python3 main.py

echo "✅ Full Setup abgeschlossen!"

