### Sample 1

Input:
```
What diffusion models have been trained on the Oxford Flower dataset?
```
Output:
```
Model #1 - Generative-Oxford-102-Flower_v2:

* Model ID: Generative-Oxford-102-Flower_v2
* File Size: 78491
* Created On: 2025-04-04T21:50:47.672074
* Last Modified: 2025-04-04T21:50:47.672074
* Framework: PyTorch 2.7
* Architecture: ConditionalUNet

The code combines features of a convolutional UNet and a conditional denoising diffusion model. The architecture consists of three main parts - the encoder, the decoder, and the conditioning layers which are used for learning class embeddings.

* Dataset: Oxford-102 Flower Dataset (Conditional Improved v3)
* Training Configuration:
	+ Batch Size: 64
	+ Learning Rate: 0.001
	+ Optimizer: Adam
	+ Epochs: 10
	+ Hardware Used: CPU
* Description: Navigate to the data folder's path (specified in script), download Oxford-102 Flowers dataset from its official website or any other available source with high quality images of 256x256 resolution saved as .png files. Saving Artifacts - The trained autoencoder and conditional denoising diffusion network are saved at the completion of each epoch using a checkpoint callback function within PyTorch's learning algorithm. Clone this GitHub repository or copy the entire codebase into your working environment to have the necessary scripts, data folders and models to start training and testing this specific example: [https://github.com/Your-Username/Conditional-Autoencoder-for-Oxford-102-Flower](https://github.com/Your-Username/Conditional-Autoencoder-for-Oxford-102-Flower)
* The script uses TensorBoard and Web-based logging frameworks (Matplotlib, PyPlot) to visualize training progress and evaluate model performance metrics:
	+ Time Embedding Visualization - Displays time embeddings for class features during a specific growth stage or lighting condition.
	+ Data Pipeline and Preprocessing
	+ Training Configuration
	+ Evaluation Methodology
	
* Purpose and Overview: The primary goal of this script is to build a conditional generative autoencoder (CGAE) for the Oxford Flowers dataset, capable of learning class embeddings and reconstructing input images with high accuracy while incorporating perceptual loss during training. This model architecture aims to be suitable for generating flower images under various conditions or lighting stages by using conditioning layers that learn features specific to each stage or condition throughout its training process.
* Test various extensions by adding new datasets or modifying existing architecture configurations such as batch normalization, dropout layers, etc., in order to further improve the performance of this CGAE model on different flower classification tasks.

+------+-----------------+------------------+---------------------+--------+------------+------------+-----------------+----------------------+-----------+------------+------------+-------+------+-----------+--------+-----+
| Rank | Model ID        | Similarity Score | Similarity Distance |   Size | Created    | Modified   | Path            | Description          | Framework | Arch       | Dataset    | Batch | LR   | Optimizer | Epochs | HW  |
+------+-----------------+------------------+---------------------+--------+------------+------------+-----------------+----------------------+-----------+------------+------------+-------+------+-----------+--------+-----+
|  1   | Generative-Oxfo |            0.235 |               0.174 | 76.7KB | 2025-04-04 | 2025-04-04 | /Users/yi-nungy | Navigate to the data | PyTorch 2 | Conditiona | Oxford-102 | 64    | 1e-0 | Adam      | 10     | CPU |
|      | rd-102-Flower_v |                  |                     |        |            |            | eh/PyCharmMiscP | folder's path        |           | lUNet      | Flower     |       | 3    |           |        |     |
|      | 2               |                  |                     |        |            |            | roject/Generati | (specified in        |           |            | Dataset (C |       |      |           |        |     |
|      |                 |                  |                     |        |            |            | ve-Oxford-102-F | script), download    |           | The code   | onditional |       |      |           |        |     |
|      |                 |                  |                     |        |            |            | lower/v2.py     | Oxford-102 Flowers   |           | combines   | Improved   |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | dataset from its     |           | features   | v3)        |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | official website or  |           | of a convo |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | any other available  |           | lutional   |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | source with high     |           | UNet and a |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | quality images of    |           | conditiona |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | 256x256 resolution   |           | l          |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | saved as .png files. |           | denoising  |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | Saving Artifacts -   |           | diffusion  |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | The trained          |           | model.     |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | autoencoder and      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | conditional          |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | denoising diffusion  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | network are saved at |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | the completion of    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | each epoch using a   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | checkpoint callback  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | function within      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | PyTorch's learning   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | algorithm. Clone     |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | this GitHub          |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | repository or copy   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | the entire codebase  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | into your working    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | environment to have  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | the necessary        |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | scripts, data        |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | folders and models   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | to start training    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | and testing this     |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | specific example: [h |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | ttps://github.com/Yo |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | ur-Username/Conditio |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | nal-Autoencoder-for- |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | Oxford-102-Flower](h |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | ttps://github.com/Yo |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | ur-Username/Conditio |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | nal-Autoencoder-for- |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | Oxford-102-Flower)   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | 5. The AST           |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | highlights the       |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | architecture of a    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | conditional color    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | classification and   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | denoising            |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | autoencoder system   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | trained on the       |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | Oxford-102 Flowers   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | dataset. Execute the |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | script using a       |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | Jupyter notebook or  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | Python command line  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | interface: this will |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | train and evaluate   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | models on the        |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | dataset while        |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | providing            |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | visualizations like  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | loss curves,         |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | tensorboard plots,   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | image samples of     |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | flowers and their    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | color labels to help |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | understand model     |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | performance          |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | effectively.         |           |            |            |       |      |           |        |     |
+------+-----------------+------------------+---------------------+--------+------------+------------+-----------------+----------------------+-----------+------------+------------+-------+------+-----------+--------+-----+
|  2   | Generative-Oxfo |            0.218 |               0.288 | 63.1KB | 2025-04-04 | 2025-04-04 | /Users/yi-nungy | Data Pipeline and    | PyTorch 2 | Conditiona | Oxford-102 | 256   | 1e-0 | Adam      | 1000   | GPU |
|      | rd-102-Flower_v |                  |                     |        |            |            | eh/PyCharmMiscP | Preprocessing**      |           | lUNet      | Flowers    |       | 3    |           |        |     |
|      | 1               |                  |                     |        |            |            | roject/Generati |                      |           |            | Dataset    |       |      |           |        |     |
|      |                 |                  |                     |        |            |            | ve-Oxford-102-F | The dataset consists |           | The code   |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            | lower/v1.py     | of the Oxford        |           | defines a  |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | Flowers              |           | Conditiona |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | (flowers_train) and  |           | lUNEt      |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | Flowers Test         |           | class,     |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | datasets for         |           | which      |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | training and         |           | combines a |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | testing,             |           | UNet archi |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | respectively.        |           | tecture    |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | Evaluation           |           | with       |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | Methodology**        |           | additional |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 |                      |           | conditioni |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | The script uses      |           | ng layers. |            |       |      |           |        |     |
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
|      |                 |                  |                     |        |            |            |                 | condition. Purpose   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | and Overview**       |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 |                      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | The primary goal of  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | this script is to    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | build a conditional  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | generative           |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | autoencoder (CGAE)   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | for the Oxford       |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | Flowers dataset,     |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | capable of learning  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | class embeddings and |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | reconstructing input |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | images with high     |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | accuracy while       |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | incorporating        |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | perceptual loss      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | during training.     |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | Model Architecture** |           |            |            |       |      |           |        |     |
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
|      |                 |                  |                     |        |            |            |                 | Test various         |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | extensions by adding |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | new datasets or      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | modifying existing   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | architecture         |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | configurations such  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | as batch             |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | normalization,       |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |        |            |            |                 | dropout layers, etc. |           |            |            |       |      |           |        |     |
+------+-----------------+------------------+---------------------+--------+------------+------------+-----------------+----------------------+-----------+------------+------------+-------+------+-----------+--------+-----+
```
