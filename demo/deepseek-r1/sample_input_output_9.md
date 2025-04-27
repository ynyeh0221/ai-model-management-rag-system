### Sample 9

Input:

```
Please find models created in March 2025 and using RNN
```

Output:

```
<think>
Okay, so I need to find models created in March 2025 that use RNN and are named something like "Multiplication_scriptRNN..." or similar. Let me start by looking at the search results provided.

First, there's Model #1: Multiplication_scriptRNN_ReversedInputString. It was created on March 3, 2025, which is after my target date of March 2025, so it might be too late. The architecture is a sequence-to-sequence model using an encoder-decoder setup with PyTorch 2.7. It uses RNN layers and has a final dense layer for prediction. That sounds relevant since the user query specifically asked for RNN models.

Next, Model #2: Subtraction_scriptRNN_ReversedInputString. Created on March 3 as well. Its architecture is just an RNN model with embedding, hidden layers, dropout, and linear output. The dataset is a math dataset, which includes subtraction problems. This might also be relevant because it uses RNNs for numerical data processing.

Model #3: Addition_scriptRNN. Created on March 2, which is within the target period. It's a Seq2Seq model with an encoder-decoder architecture using PyTorch 2.7 and RNN layers. The dataset is again math-based, focusing on addition problems.

Wait, all three models are created in or after March 2025, so they fit the criteria. Model #1 deals with multiplication but uses a reversed input string approach. Models #2 and #3 handle subtraction and addition respectively, using standard RNN setups within PyTorch.

I should note that while these models use RNNs, Model #1 has a slightly different architecture since it's an encoder-decoder for sequence translation, whereas the others are simpler RNNs used for specific math operations like subtraction and addition. 

The user might be interested in comparing how RNNs can be applied to different arithmetic operations or perhaps looking at variations in model architectures. Since all three are recent, they might represent up-to-date implementations that could serve as references.

I should present each model individually, highlighting their names, creation dates, frameworks, architectures, datasets, and any specific details about their implementation. This way, the user can see which ones are suitable for their needs based on the operation type (addition, subtraction) and architecture style.
</think>

Based on your query focusing on RNN models created in March 2025 that handle arithmetic operations like multiplication, addition, or subtraction, here's a structured presentation of the relevant models:

### Model #1: Multiplication_scriptRNN_ReversedInputString
- **Created Date**: March 3, 2025
- **Framework**: PyTorch 2.7
- **Architecture**: Sequence-to-Sequence (Seq2Seq) model with an encoder-decoder setup.
- **Components**:
  - Encoder: Uses RNN layers with dropout to process input sequences.
  - Decoder: Inheriting from `RNNModel` for output generation.
  - Final Dense Layer: Produces predictions using activation functions.
- **Dataset**: Focuses on multiplication problems with reversed input strings.

### Model #2: Subtraction_scriptRNN_ReversedInputString
- **Created Date**: March 3, 2025
- **Framework**: PyTorch 2.7
- **Architecture**: Simple RNN model.
- **Components**:
  - Tokenizer for numerical tokens.
  - Embedding layer for converting tokens to dense vectors.
  - Three RNN layers with dropout.
  - Linear output layer with activation function.
- **Dataset**: Subtraction-based samples generated.

### Model #3: Addition_scriptRNN
- **Created Date**: March 2, 2025
- **Framework**: PyTorch 2.7
- **Architecture**: Seq2SeqModel with encoder-decoder structure.
- **Components**:
  - Encoder and decoder RNN layers for processing sequences.
  - Embedding layer mapping tokens to latent space vectors (size 256).
  - Two fully connected layers followed by a softmax activation for predictions.
- **Dataset**: Math problems focusing on addition.

### Key Points:
- All three models were created in or after March 2025, fitting your temporal criteria.
- Models #1 and #3 use an encoder-decoder architecture (with Model #1 specialized for multiplication), while Model #2 is a simpler RNN setup.
- Each model handles different arithmetic operations: multiplication with reversed inputs, subtraction, and addition.

These models provide a range of examples for applying RNNs to numerical problem-solving tasks.

Retrieving and displaying reranked search results:

+------+-----------------+------------------+---------------------+-------+------------+------------+-----------------+----------------------+-----------+------------+------------+-------+------+-----------+--------+-----+
| Rank | Model ID        | Similarity Score | Similarity Distance |  Size | Created    | Modified   | Path            | Description          | Framework | Arch       | Dataset    | Batch | LR   | Optimizer | Epochs | HW  |
+------+-----------------+------------------+---------------------+-------+------------+------------+-----------------+----------------------+-----------+------------+------------+-------+------+-----------+--------+-----+
|  1   | Multiplication_ |            0.170 |               0.724 | 4.4KB | 2025-03-03 | 2025-03-03 | /Users/yi-nungy | The trained model    | PyTorch 2 | Sequence-  | Multiplica | 64    | 1e-0 | Adam      | 100    | GPU |
|      | scriptRNN_Rever |                  |                     |       |            |            | eh/PyCharmMiscP | can predict the      |           | to-        | tion       |       | 3    |           |        |     |
|      | sedInputString  |                  |                     |       |            |            | roject/Calculat | result of given      |           | sequence   | dataset    |       |      |           |        |     |
|      |                 |                  |                     |       |            |            | ion/Multiplicat | input expressions,   |           | (Encoder-  |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            | ion/scriptRNN_R | with an accuracy-    |           | Decoder)   |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            | eversedInputStr | focused architecture |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            | ing.py          | suitable for         |           | The archit |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | numerical data       |           | ecture is  |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | processing tasks in  |           | inferred   |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | education or         |           | from the   |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | mathematics problem  |           | presence   |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | solving              |           | of a       |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | applications. The    |           | Tokenizer  |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | train_loader and     |           | class and  |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | test_loader handle   |           | forward    |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | data preprocessing   |           | method in  |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | steps like           |           | the        |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | tokenization,        |           | RNNModel   |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | padding, and         |           | class, as  |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | shuffling before     |           | well as    |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | feeding them to the  |           | the use of |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | RNNModel. **Model    |           | an encoder |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | Architecture**       |           | /decoder   |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 |                      |           | module     |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | The core of the      |           | within the |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | model is an RNNModel |           | model.     |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | class (inherits from |           | Also       |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | torch.nn.Module),    |           | evident    |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | which consists of a  |           | are the    |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | vocabulary size,     |           | DataLoader |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | embedding dimension, |           | instantiat |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | hidden layer         |           | ions for   |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | dimensions, number   |           | loading    |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | of layers, and       |           | data into  |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | dropout rate as its  |           | the model. |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | hyperparameters.     |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | LSTM/GRU Recurrent   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | Neural Networks      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | (RNNs) layer(s):     |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | these take input     |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | sequences and output |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | a hidden state       |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | representation,      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | learning over time   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | from the training    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | data. Dense linear   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | layer on top of the  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | last RNN's output:   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | this predicts the    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | multiplication       |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | result given the     |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | learned features.    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | The model is trained |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | for up to 100        |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | epochs, each         |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | containing multiple  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | iterations of the    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | forward() function   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | in the RNNModel's    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | training loop.       |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | However, when you    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | create a trained     |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | model with this      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | architecture, it is  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | recommended to       |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | implement plotting   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | or other methods for |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | monitoring training  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | progress, checking   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | the embedding space, |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | inspecting confusion |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | matrices or          |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | prediction results,  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | etc., as per your    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | specific use case    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | and needs.           |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | **Reproduction and   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | Extension Guide**    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 |                      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | To reproduce this    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | example, follow      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | these steps:         |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | 1. Ensure you have a |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | CUDA-capable GPU     |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | available with       |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | PyTorch installed    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | for efficient        |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | training of neural   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | networks if          |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | necessary. Clone the |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | provided GitHub      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | repository (if       |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | present) containing  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | relevant data or     |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | create your own      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | multiplications      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | samples dataset      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | according to this    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | code's structure.    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | Run the main script  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | located within this  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | cloned repository or |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | copied local copy of |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | the data to train    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | and evaluate the     |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | model using          |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | appropriate          |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | DataLoaders for      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | training and         |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | testing, then check  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | its performance      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | metrics like loss,   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | accuracy etc.,       |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | calculated by        |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | PyTorch during       |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | forward() function   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | calls. For           |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | extension, you could |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | modify               |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | hyperparameters,     |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | extend the dataset   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | size (e.g., increase |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | vocabulary), or try  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | different RNN models |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | with varying         |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | architectures to     |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | achieve better       |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | results in your      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | specific use case.   |           |            |            |       |      |           |        |     |
+------+-----------------+------------------+---------------------+-------+------------+------------+-----------------+----------------------+-----------+------------+------------+-------+------+-----------+--------+-----+
|  2   | Subtraction_scr |            0.168 |               0.572 | 4.5KB | 2025-03-03 | 2025-03-03 | /Users/yi-nungy | **Purpose and        | PyTorch 2 | RNN Model  | Math       | 64    | 1e-0 | Adam      | 100    | GPU |
|      | iptRNN_Reversed |                  |                     |       |            |            | eh/PyCharmMiscP | Overview**           |           |            | Dataset    |       | 3    |           |        |     |
|      | InputString     |                  |                     |       |            |            | roject/Calculat |                      |           | The archit |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            | ion/Subtraction | This document        |           | ecture is  |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            | /scriptRNN_Reve | describes a PyTorch- |           | a          |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            | rsedInputString | based machine        |           | Recurrent  |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            | .py             | learning model       |           | Neural     |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | script that uses an  |           | Network    |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | RNN to solve         |           | (RNN)      |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | mathematical         |           | model that |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | expression           |           | includes   |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | generation problems. |           | an         |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | The model takes in   |           | embedding  |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | numerical            |           | layer,     |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | expressions as input |           | hidden     |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | and generates the    |           | layers     |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | correct answer,      |           | with long- |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | teaching the neural  |           | short term |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | network how to       |           | memory     |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | calculate various    |           | units for  |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | types of arithmetic  |           | processing |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | operations. This     |           | sequences  |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | script utilizes      |           | of tokens, |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | DataLoader for       |           | and a      |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | efficient data       |           | final      |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | preprocessing,       |           | linear     |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | Tokenizer class for  |           | output     |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | tokenizing input     |           | layer.     |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | expressions,         |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | MathDataset class    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | for pre-processing   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | datasets, and an     |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | RNNModel class with  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | 3 hidden layers and  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | dropout              |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | regularization. The  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | model is trained on  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | a dataset of ~20k    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | samples and          |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | evaluated using the  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | test set after 100   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | epochs. This         |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | Tokenizer is then    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | utilized in the      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | `MathDataset`, which |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | contains two         |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | classes, `__init__`  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | for initializing     |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | dataset with data    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | and tokenizer and    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | `__len__` for        |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | returning the length |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | of training          |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | set/batch, and an    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | index method to      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | retrieve each sample |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | from the dataset.    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | The script generates |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | subtraction-based    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | samples using a      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | function called 'gen |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | erate_subtraction_da |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | ta'. The generated   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | datasets are then    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | used as inputs in    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | both the train and   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | test DataLoader      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | instances with batch |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | sizes of 64. **Model |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | Architecture**       |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 |                      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | The model is an RNN  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | (`RNNModel`), which  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | inherits from        |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | `Module`. The        |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | architecture         |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | includes the         |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | following            |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | components:          |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | - A Tokenizer for    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | tokenizing           |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | mathematical         |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | expressions          |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | - An embedding layer |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | that converts        |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | numerical tokens     |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | into dense vectors   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | - Three RNN layers   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | with dropout to      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | avoid overfitting    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | and learn from a     |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | diverse set of data  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | - A linear output    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | layer with           |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | activation function  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | for predicting       |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | correct answers.     |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | **Evaluation and     |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | Testing              |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | Methodology**        |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 |                      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | The script utilizes  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | an `evaluate` method |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | on the model for     |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | testing.             |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | **Reproduction and   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | Extension Guide**    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 |                      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | To reproduce this    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | implementation, run  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | the code starting    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | from loading         |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | necessary libraries  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | and dependencies     |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | (e.g., PyTorch) to   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | generating           |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | subtraction-based    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | samples with 'genera |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | te_subtraction_data' |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | function. Then       |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | utilize preprocessed |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | data for model       |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | training by passing  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | it to DataLoader     |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | instances in         |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | train\_loader and    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | test\_loader         |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | variables.           |           |            |            |       |      |           |        |     |
+------+-----------------+------------------+---------------------+-------+------------+------------+-----------------+----------------------+-----------+------------+------------+-------+------+-----------+--------+-----+
|  3   | Addition_script |            0.145 |               0.527 | 4.4KB | 2025-03-02 | 2025-03-02 | /Users/yi-nungy | To extend this       | PyTorch 2 | Seq2SeqMod | MathDatase | 64    | 1e-0 | Adam      | 100    | GPU |
|      | RNN             |                  |                     |       |            |            | eh/PyCharmMiscP | model, we could      |           | el         | t          |       | 3    |           |        |     |
|      |                 |                  |                     |       |            |            | roject/Calculat | consider exploring   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            | ion/Addition/sc | different RNN        |           | The model  |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            | riptRNN.py      | architectures or     |           | uses a     |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | additional dataset   |           | simple     |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | sources for training |           | encoder-   |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | purposes. This       |           | decoder ar |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | document is          |           | chitecture |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | comprehensive in     |           | with       |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | covering every       |           | PyTorch's  |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | component and        |           | RNNModel   |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | process involved in  |           | class,     |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | this token-based     |           | indicating |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | math problem solving |           | a Seq2Seq  |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | model, ensuring it   |           | task.      |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | meets the quality    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | verification         |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | checklist's criteria |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | before finalizing    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | the documentation.   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | **Purpose and        |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | Overview**           |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 |                      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | The provided code    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | implements a simple  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | language modeling    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | problem, where the   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | goal is to predict   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | the next token in a  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | sequence of input    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | tokens. The model    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | uses a recurrent     |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | neural network (RNN) |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | with a bidirectional |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | Long Short-Term      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | Memory (LSTM) layer  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | to handle sequential |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | data. The trained    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | model can then be    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | used for inference   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | on new addition data |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | sequences. **Data    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | Pipeline and         |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | Preprocessing**      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 |                      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | The code starts by   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | importing the        |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | required libraries,  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | including PyTorch    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | for deep learning    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | framework and        |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | DataLoader for       |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | loading datasets in  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | batches. Then it     |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | inherits a Tokenizer |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | class from           |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | torch.utils.data to  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | perform tokenization |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | of input expressions |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | and decoding tokens. |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | The collate_fn       |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | function handles     |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | batching and sorting |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | inputs for           |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | DataLoader. A genera |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | te_addition_data()   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | function generates   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | 20k random addition  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | problems for         |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | training, testing,   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | and validation sets. |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | **Model              |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | Architecture**       |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 |                      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | The model            |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | architecture         |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | consists of an       |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | embedding layer that |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | maps input tokens to |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | vectors in a latent  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | space with size 256. |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | Finally, the         |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | concatenated outputs |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | from each direction  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | are processed by two |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | fully connected (FC) |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | layers before making |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | predictions using    |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | softmax activation.  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | **Evaluation and     |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | Testing              |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | Methodology**        |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 |                      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | The trained model's  |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | performance is       |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | evaluated by calling |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | the forward()        |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | function inside      |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | evaluate() method,   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | which predicts the   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | next token in each   |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | input sequence up to |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | a specified          |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | max_length           |           |            |            |       |      |           |        |     |
|      |                 |                  |                     |       |            |            |                 | (defaulted as 10).   |           |            |            |       |      |           |        |     |
+------+-----------------+------------------+---------------------+-------+------------+------------+-----------------+----------------------+-----------+------------+------------+-------+------+-----------+--------+-----+
```
