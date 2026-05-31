## Welcome to the Pytorch Implementation of "ActMVS: Active Scene Reconstruction with Monocular Multi-View Stereo" for ICRA 2026.

## 0.Setup
Create Conda Env
```
conda create -y -n ActMVS python=3.9 cmake=3.14.0
conda activate ActMVS

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/liren-jin/diff-gaussian-rasterization_2d
pip install git+https://github.com/nianticlabs/mvsanywhere

cd simulator
git clone git@github.com:liren-jin/habitat-sim.git
cd habitat-sim
pip install -r requirements.txt
python setup.py install --headless --bullet
```
Download Replica Data
```
bash scripts/replica_download.sh
```

## 1.Run
#### make sure that "mesh_path" and "scene_id" in "config.yaml" are pointing to your downloaded replica data.
```
bash scripts/run.sh
```
Feel free to change hyperparameters in "config.yaml". As default, "budget: 600" sets the total exploration duration to 600 seconds (increase for more completion).

## 2.Check Results
#### Intermediate visualizations during active reconstruction:
<div align="left">
  <img src="results/Reconstruction_Vis.png" alt="Reconstruction_Vis" width="800">
</div>

From left to right: Input Frame, Predicted Depth, Rendered Mask, Rendered Frame, Rendered Depth, Rendered Normal, Rendered Opacity.

#### Intermediate Point Clouds:
<div align="left">
  <img src="results/PCD_vis.png" alt="PCD_vis" width="400">
</div>

#### Final Mesh:
<div align="left">
  <img src="results/Mesh_vis.png" alt="Mesh_vis" width="400">
</div>
Point clouds and Mesh are stored as ".ply" files and can be viewed in MeshLab.

## Thanks for your attention! More instructions will be added soon!
