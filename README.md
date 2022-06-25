High Resolution Image Classification with Rich Text Information Based on Graph Convolution Neural Network
============

- Paper link: www.xxx.com
- Author's code repo: www.xxx.com

Dependencies
------------
- PyTorch 1.1.0+
- tensorboard
- tensorboardX
- dgl 0.8.1
- argparse
- tqdm

Downloads
----------

- https://pan.baidu.com/s/1s1z_2lnqnRdRbtE_WfzOzQ?pwd=ahd7
- https://drive.google.com/drive/folders/1UmNaKLqPIQi2rThbTkFOFTtyUeVaCdty?usp=sharing

First, you should download the dataset and backbones from either of the links above. Put the two folders into the root path. Unzip images.zip into ./data/ and unzip graph.bin.zip into ./data/graph.

How to run
----------

- Baseline Models

If you want to run the CNN baselines, you should run the file in ./data/ to construct the data for CNNs.

```bash
python generate_cnn_dataset.py
```

Then, you can run the following codes to train ResNet18, GoogLeNet, and VGG19.

```bash
python baseline_ResNet.py
```

```bash
python baseline_GoogLeNet.py
```

```bash
python baseline_VGG.py
```

- Node-based Classification Model




- Graph-based Classification Model









