# CS6910_Assignment_3
### Rishabh Kawediya CS22M072
[Report Link](https://wandb.ai/cs22m072/DLAssignment3/reports/CS6910-Assignment-3--Vmlldzo0NDE4ODYw?accessToken=od7cb5nl96oucpyc60k1hi8obagcp71pgm3613pcapvl2dv4cinau2w2ydcu8pa0) <br>
Assignment3 contains code for all questions <br>
## **Packages**
- wandb
- torch
- pandas

## **train.py**
To evaluate vanilla encoder decoder architecture

## **train_a.py**
To evaluate attention based encoder decoder architecture

## **Classes and Functions**
- Encoder class
    - forward 
    - Perform encoder cell operation
    - init hidden
    - To initialize hidden and cell tensors

- Decoder class
    - forward 
    - Perform decoder cell operation
    - init hidden
    - To initialize hidden and cell tensors

- Seq2Seq class
    - forward 
    - To connect encoder and decoder
- train function 
    - To train the model
- train_accuracy function
    - To calculate the accuracy on train data
- accuracy function
    - To calculate the accuracy on validation data

## **To evaluate**

```
python train.py --wandb_entity myname --wandb_project myprojectname
```
```
python train_a.py --wandb_entity myname --wandb_project myprojectname
```

| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-wp`, `--wandb_project` | myprojectname | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | myname  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `-e`, `--epochs` | 1 |  Number of epochs to train neural network.|
| `-b`, `--batch_size` | 4 | Batch size used to train neural network. | 
| `-ct`, `--cell_type` | "lstm" | choices:  ["lstm","gru","rnn"] |
| `-es`, `--embedding_size` | 512 | embedding layer dimension | 
| `-hs`, `--hidden_size` | 512 | Hidden cell dimension | 
| `-nl`, `--num_layers` | 3 | Num of layers used by cells. |
| `-d`, `--dropout` | 0.2 | Dropout used by embedding and cells |
| `-bd`, `--bidirectional` | "Yes" | If cells should be bidirectional or not |
<br>


