#!/bin/bash
set -e

echo "🚀 NVIDIA GPU Environment Fixer - Workstation Edition (v3)"

# 1️⃣ Systeminfo
echo "🔎 Systeminfo:"
lsb_release -a || cat /etc/os-release
uname -r
echo ""

# 2️⃣ Cleanup
echo "🧹 Alte NVIDIA Pakete entfernen..."
sudo rm -f /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get purge -y '^nvidia-.*' '^libnvidia-.*' 'cuda-.*' 'nvidia-container.*' || true
sudo apt-get autoremove -y
sudo apt-get autoclean -y

# 3️⃣ NVIDIA Treiber installieren
echo "💾 NVIDIA Treiber (550) installieren..."
sudo add-apt-repository -y ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install -y nvidia-driver-550

# 4️⃣ NVIDIA Container Toolkit direkt aus GitHub (aktuellste Version)
echo "💾 NVIDIA Container Toolkit direkt aus GitHub beziehen..."
mkdir -p /tmp/nvidia-container-toolkit
cd /tmp/nvidia-container-toolkit

# Automatisch die aktuellen URLs auslesen
BASE_URL="https://nvidia.github.io/libnvidia-container/stable/deb/amd64/"
for PACKAGE in libnvidia-container1 libnvidia-container-tools nvidia-container-toolkit; do
    PACKAGE_URL=$(curl -s $BASE_URL | grep -oE "$PACKAGE.*\.deb" | head -n 1)
    if [ -z "$PACKAGE_URL" ]; then
        echo "❌ Fehler: Konnte $PACKAGE nicht finden!"
        exit 1
    fi
    wget "$BASE_URL$PACKAGE_URL"
done

# Installieren
sudo dpkg -i *.deb

# 5️⃣ Docker Runtime sicherstellen
echo "⚙️ Docker Runtime konfigurieren..."
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF
sudo systemctl restart docker

# 6️⃣ Reboot-Hinweis
echo "🔄 Reboot dringend empfohlen!"
echo ""

# 7️⃣ Testlauf
echo "🧪 Test: Docker GPU Zugriff"
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

echo "✅ Fix abgeschlossen!"
