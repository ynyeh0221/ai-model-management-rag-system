### Sample 2

Input:
```
Which models use Unet architectures?
```

Output:
```
### EXECUTIVE SUMMARY

This report presents a detailed analysis of three models that use variations of the Unet architecture: Generative-Fashion-MNIST_script_unet, Generative-Oxford-102-Flower_v1, and Generative-Fashion-MNIST_latent_new_model. These models employ UNet or ConditionalUnet architectures for various tasks such as image denoising, flower classification, and fashion item recognition using grayscale images. The three models have distinct features, hyperparameters, frameworks, and datasets.

### TECHNICAL SPECIFICATIONS

Model 1: Generative-Fashion-MNIST_latent_new_model
* Architecture: SimpleUNet
	+ Input channels: Equal to 1
	+ Hidden dimensions: [32, 64, 128]
	+ Residual blocks: UNetResidualBlock (5)
	* Framework: PyTorch 2.7
* Dataset: Fashion MNIST
* Training Configuration:
	+ Batch Size: 256
	+ Learning Rate: 0.001
	+ Optimizer: Adam
	+ Epochs: 10
	+ Hardware Used: GPU (or CPU)

Model 2: Generative-Oxford-102-Flower_v1
* Architecture: ConditionalUNet
	+ Decoder with conditioning layers for flower classification
	* Dataset: Oxford-102 Flowers Dataset
* Framework: PyTorch 2.7
* Training Configuration:
	+ Batch Size: 256
	+ Learning Rate: 0.001
	+ Optimizer: Adam
	+ Epochs: 1000
	+ Hardware Used: GPU (or CPU)

Model 3: Generative-Fashion-MNIST_script_unet
* Architecture: UNetDenoiser
	+ Modified U-Net architecture for grayscale image denoising
	* Dataset: Fashion MNIST
* Framework: PyTorch 2.7
* Training Configuration:
	+ Batch Size: 128
	+ Learning Rate: 0.0002
	+ Optimizer: AdamW
	+ Epochs: 150
	+ Hardware Used: CPU (or GPU)

### IMPLEMENTATION DETAILS

Each model has its distinct implementation details, such as the number of layers and parameters in UNet or ConditionalUNet architectures. The SimpleAutoencoder class in Model 1 employs the aforementioned architecture with a few adjustments for autoencoding tasks. Similarly, conditional branches are included in the ConditionalUNet architecture used in Model 2 to handle flower classification.

### PERFORMANCE ANALYSIS

Each model offers different performance metrics based on their respective architectures and datasets. The Generative-Fashion-MNIST_latent_new_model achieves better denoising results, while the other two models excel in classifying flowers using the Fashion MNIST dataset.

### TECHNICAL INSIGHTS

These three UNet or ConditionalUnet variations showcase how a well-structured neural network architecture can be adapted to address various tasks like image denoising and classification by incorporating additional conditioning layers in conditional networks. However, further enhancements could include adjusting hyperparameters for better learning, implementing data augmentation techniques to increase training diversity, and fine-tuning the UNet architecture based on certain performance criteria (e.g., minimal amount of noise).

+------+-----------------+------------------+---------------------+--------+------------+------------+-----------------+----------------------+-----------+------------+------------+-------+------+-----------+--------+-----+
| Rank | Model ID        | Similarity Score | Similarity Distance |   Size | Created    | Modified   | Path            | Description          | Framework | Arch       | Dataset    | Batch | LR   | Optimizer | Epochs | HW  |
+------+-----------------+------------------+---------------------+--------+------------+------------+-----------------+----------------------+-----------+------------+------------+-------+------+-----------+--------+-----+
|  1   | Generative-     |            0.184 |               0.716 | 43.4KB | 2025-04-09 | 2025-03-16 | /Users/yi-nungy | The model uses the   | PyTorch 2 | SimpleUNet | Fashion    | 256   | 1e-0 | Adam      | 10     | GPU |
|      | Fashion-MNIST_l |                  |                     |        |            |            | eh/PyCharmMiscP | UNet architecture's  |           |            | MNIST      |       | 3    |           |        |     |
|      | atent_new_model |                  |                     |        |            |            | roject/Generati | final latent space   |           | The script |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            | ve-Fashion-MNIS | representation to    |           | contains i |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            | T/latent_new_mo | evaluate its         |           | mplementat |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            | del.py          | performance by       |           | ions of    |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | computing these      |           | UNET and U |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | metrics on test sets |           | NetAttenti |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | using                |           | onBlock    |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | `TestDataLoader`.    |           | classes,   |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | `SimpleUNet`: A UNet |           | along with |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | architecture with    |           | other      |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | input channels equal |           | related    |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | to 1, hidden         |           | modules    |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | dimensions [32, 64,  |           | such as    |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | 128], and five       |           | Encoder,   |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | residual blocks (`Un |           | Decoder,   |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | etResidualBlock`).   |           | and UNetRe |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | The model is trained |           | sidualBloc |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | using the Adam       |           | k. Additio |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | optimizer with       |           | nally, it  |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | learning rate        |           | features S |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | LR=0.0002 and weight |           | impleAutoe |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | decay WD=0.0005 on a |           | ncoder     |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | device of either CPU |           | class      |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | or NVIDIA GPU        |           | which      |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | depending on their   |           | inherits   |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | availability         |           | from       |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | (`torch.device('mps' |           | Module.    |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | if torch.mps.is_avai |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | lable() else         |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | 'cpu')`). Training   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | Configuration**      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | Training             |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | configuration        |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | includes the         |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | following            |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | parameters:          |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | - `optimizer`: The   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | Adam optimizer is    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | used for all models  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | with initial         |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | learning rates set   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | to LR=0.0002 and     |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | weight decay of      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | WD=0.0005. - `loss`: |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | PyTorch's loss       |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | module is used for   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | calculating losses   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | during training,     |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | including            |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | contrastive loss for |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | autoencoder training |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | and Focal Losses     |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | with Softplus        |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | activation functions |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | to handle unbalanced |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | datasets in          |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | denoising diffusion  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | processes (Fashion   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | MNIST).              |           |            |            |       |      |           |        |     |
+------+-----------------+------------------+---------------------+--------+------------+------------+-----------------+----------------------+-----------+------------+------------+-------+------+-----------+--------+-----+
|  2   | Generative-Oxfo |            0.154 |               0.768 | 63.1KB | 2025-04-04 | 2025-04-04 | /Users/yi-nungy | The decoder is an    | PyTorch 2 | Conditiona | Oxford-102 | 256   | 1e-0 | Adam      | 1000   | GPU |
|      | rd-102-Flower_v |                  |                     |        |            |            | eh/PyCharmMiscP | UNet architecture    |           | lUNet      | Flowers    |       | 3    |           |        |     |
|      | 1               |                  |                     |        |            |            | roject/Generati | with several custom  |           |            | Dataset    |       |      |           |        |     |
|      |                 |                  |                     |        |            |            | ve-Oxford-102-F | modules such as:     |           | The code   |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            | lower/v1.py     |                      |           | defines a  |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | - UNetAttentionBlock |           | Conditiona |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | (inherits from       |           | lUNEt      |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | Sequential), which   |           | class,     |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | adds a skip          |           | which      |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | connection between a |           | combines a |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | convolution layer    |           | UNet archi |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | and a transposed     |           | tecture    |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | convolution layer    |           | with       |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | for residual         |           | additional |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | connections within   |           | conditioni |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | each UNet block.     |           | ng layers. |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | Evaluation           |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | Methodology**        |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 |                      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | The script uses      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | TensorBoard and Web- |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | based logging        |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | frameworks           |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | (Matplotlib, PyPlot) |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | to visualize         |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | training progress    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | and evaluate model   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | performance metrics: |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 |                      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | - Time Embedding     |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | Visualization -      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | Displays time        |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | embeddings for class |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | features during a    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | specific growth      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | stage or lighting    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | condition. Training  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | Methodology**        |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 |                      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | The script uses      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | TensorFlow's Keras   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | API for deep         |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | learning model       |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | construction and the |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | PyTorch framework to |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | handle GPU           |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | computations when    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | available. Model     |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | Architecture**       |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 |                      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | The primary neural   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | network model used   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | in this script is    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | the                  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | SimpleAutoencoder    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | class, which         |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | consists of two      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | encoder and decoder  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | modules with         |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | multiple custom      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | layers, including:   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 |                      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | - Swish module       |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | (inherits from       |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | Module), an          |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | activation function  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | for replacing ReLU.  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | Training             |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | Configuration**      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 |                      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | The training process |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | consists of several  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | optimizers, learning |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | rates, loss          |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | functions, and       |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | evaluation metrics:  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 |                      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | - Optimizer -        |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | Stochastic Gradient  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | Descent (Adam) with  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | L2 regularization on |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | weights.             |           |            |            |       |      |           |        |     |
+------+-----------------+------------------+---------------------+--------+------------+------------+-----------------+----------------------+-----------+------------+------------+-------+------+-----------+--------+-----+
|  3   | Generative-     |            0.141 |               0.225 | 33.8KB | 2025-03-14 | 2025-03-14 | /Users/yi-nungy | However, some areas  | PyTorch 2 | UnetDenois | Fashion    | 128   | 2e-0 | AdamW     | 150    | CPU |
|      | Fashion-MNIST_s |                  |                     |        |            |            | eh/PyCharmMiscP | that could improve   |           | er         | MNIST      |       | 4    |           |        |     |
|      | cript_unet      |                  |                     |        |            |            | roject/Generati | or extend this model |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            | ve-Fashion-MNIS | include adjusting    |           | The model  |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            | T/script_unet.p | hyperparameters for  |           | is a       |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            | y               | better learning,     |           | modified   |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | implementing data    |           | U-Net arch |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | augmentation         |           | itecture s |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | techniques to        |           | pecificall |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | increase training    |           | y designed |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | diversity, and fine- |           | for        |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | tuning the UNet      |           | denoising  |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | architecture based   |           | Fashion    |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | on certain denoising |           | MNIST      |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | criteria (e.g.,      |           | images. It |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | minimal amount of    |           | consists   |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | noise). The script   |           | of several |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | ends with a device   |           | residual   |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | setting based on     |           | blocks,    |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | whether GPU or MPS   |           | patch conv |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | (if available)       |           | olutional  |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | should be used for   |           | layers,    |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | computation, and     |           | and a      |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | then loads the Fashi |           | self-      |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | onMNISTUNetDenoiser  |           | attention  |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | class onto it which  |           | mechanism. |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | is initialized using |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | certain parameters   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | such as image size,  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | input channels,      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | number of output     |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | classes, time        |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | dimension, and patch |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | size. Lastly, the    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | optimizer AdamW with |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | a learning rate at   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | 0.0002, weight decay |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | at 0.02, and betas   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | at (0.9, 0.99) is    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | assigned to train    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | this model through   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | epochs. FashionMNIST |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | UNetDenoiser class   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | inherits from        |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | Module, which        |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | provides the overall |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | structure of the     |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | U-Net architecture   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | for denoising        |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | grayscale images     |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | using a time         |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | dimension and patch  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | size parameters. It  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | uses PyTorch as the  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | main framework, and  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | trains a model on a  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | dataset containing   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | 60,000 grayscale     |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | images of fashion    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | items.               |           |            |            |       |      |           |        |     |
+------+-----------------+------------------+---------------------+--------+------------+------------+-----------------+----------------------+-----------+------------+------------+-------+------+-----------+--------+-----+
```
