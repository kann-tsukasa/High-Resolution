High Resolution Image Classification with Rich Text Information Based on Graph Convolution Neural Network
============

- Paper link: www.xxx.com
- Author's code repo: www.xxx.com

1.Dependencies
------------
- PyTorch 1.1.0+
- tensorboard
- tensorboardX
- dgl 0.8.1
- argparse
- tqdm

2.Downloads
----------

### Links

- https://pan.baidu.com/s/1s1z_2lnqnRdRbtE_WfzOzQ?pwd=ahd7
- https://drive.google.com/drive/folders/1UmNaKLqPIQi2rThbTkFOFTtyUeVaCdty?usp=sharing

First, you should download the dataset and backbones from either of the links above. Put the two folders into the root path. Unzip images.zip into ./data/ and unzip graph.bin.zip into ./data/graph.

3.How to run
----------

### 3.1 Baseline Models

If you want to run the CNN baselines, you should run the python file in ./data/ to construct the data for CNNs.

```bash
python generate_cnn_dataset.py
```

Then, you can run the following codes to train ResNet18, GoogLeNet, and VGG19. Set --show 1 to show the progress bar. If you want to use the nohup command, you should set --show 0. 

```bash
python baseline_ResNet.py --show 1
```

```bash
python baseline_GoogLeNet.py --show 1
```

```bash
python baseline_VGG.py --show 1
```

### 3.2 GNN-based Models

To run the NBCM and GBCM, we should generate the graph data from the original image data. Of course, we have already provided off-the-shelf graph
data in ./data/graph/ and ./data/graphs, so that this step could be skipped.  

First, use CRAFT to generate the sub-images.

```bash
python generatesubimgs.py
```
Then, construct the BFS-based graphs for the sub-images.

```bash
python buildbfsgraph.py
```

#### 3.2.1 Node-based Classification Model

Construct the graph for NBCM.

```bash
python constructgraph.py
```

Train NBCM. Set --pretrain 1 to load the pre-trained weights for the ResNet18 in NBCM. 

```bash
python nbcm.py
```

#### 3.2.2 Graph-based Classification Model

Use STR model to generate the high dimension features for each node and construct graphs for GBCM.


```bash
python constructgraphs.py
```

Train GBCM. If you want to use the nohup command, you should set --show 0. 

```bash
python .py --show 1
```
