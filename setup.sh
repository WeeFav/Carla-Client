wget https://tiny.carla.org/carla-0-9-15-linux
tar -xvzf carla-0-9-15-linux -C CARLA_0.9.15

wget -O carla-0.9.15.whl https://files.pythonhosted.org/packages/86/5a/74cf6657e5bb4b8ffe790ef92a01d7f44592a992d758c7053cd6bc75bd96/carla-0.9.15-cp310-cp310-manylinux_2_27_x86_64.whl

source /opt/miniforge3/etc/profile.d/conda.sh

conda create -y -n carla python=3.10
conda activate carla
pip install pygame numpy opencv-python pillow open3d
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install carla-0.9.15.whl

wget -O vscode.deb https://code.visualstudio.com/sha/download?build=stable&os=linux-deb-x64
sudo apt install vscode.deb