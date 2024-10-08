<h1>Generation Depth V2</h1>
<h2>Depth Map Generation Using ZoeDepth / Depth-AnythingV2 And Tiling</h2>
  
[**Random-Code-Guy**]

**Project Started 19/07/2024**

This project started out as a hobby to try generating depth maps using ZeoDepth but
then expanded to trying to use tiling to increese the depth map quality

## Install Requirements
```
pip install customtkinter
sudo apt-get install python3-tk
pip install numpy==1.26.4
pip install opencv-python
pip install matplotlib
pip install torch==2.3.1+cpu
pip install torchvision==0.18.1+cpu
pip install Pillow
pip install cython
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```

## GUI Application

This application has a very simple gui using tkinter with generation done in its own
thread to keep it stable

<img src='https://iili.io/dxXnd0X.png'>

This is a work in progress please feel free to submit code to this project i will test and add everything that
passes tests and improves the application

## Pre-trained Models Install

**four** Pre-trained models of varying scales for robust relative depth estimation Can be found in the depth anything v2 hugging face
download these models and save then into the models folder created by the application after first launch

| Model | Params | Checkpoint |
|:-|-:|:-:|
| Depth-Anything-V2-Small | 24.8M | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true) |
| Depth-Anything-V2-Base | 97.5M | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true) |
| Depth-Anything-V2-Large | 335.3M | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true) |
| Depth-Anything-V2-Giant | 1.3B | Coming soon |

