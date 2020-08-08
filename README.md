# DeepImageRanking
Image Similarity using Deep Ranking

## Documentation
1. For this particular implementation of Deep Ranking we are using TinyImageNet as our training dataset. Download the TinyImageNet dataset [here](https://tiny-imagenet.herokuapp.com/)
2. Change the folder structure of the validation set by running:
    ```bash
    python transform_validation.py
    ```
    This will aid later when generating the image embeddings
3. Now to sample triplets we run:
    ```bash
    python triplet_sampler.py --training 1 --num_pos_images 1 --num_neg_images 1
    ```
    Our command-line arugments are as follows:
      * training: 0 or 1 to sample from train or validation set
      * num_pos_images: the number of positive images per query image
      * num_neg_images: the number of negative images per query image

    This should generate `training_triplet_sample.csv` or `val_triplet_sample.csv` based on the argument provided.

2. To implement we use a pre-trained ResNet-101 with two sub-sampling layers. We finetune the earlier layers of the ResNet and train the rest. The PyTorch implementation can be found in the file `deep_rank_net.py`
4. To train the network run:
    ```bash
     python train_net.py --epochs 1 --optim sgd
    ```
    Our command-line arugments are as follows:
    * epochs: the number of training epochs to run
    * optim: thee optimizer to use, can be `sgd` or `adam` or `rms`. Will default to `sgd`.

   Other hyperparameters are set inside the file. These include `BATCH_SIZE=25` and `LEARNING_RATE=0.001`
   This should generate a fully trained model file called `deepranknet.model` and other intermediate files called `temp_*.model`
5. To generate embeddings:
    ```bash
    python gen_emebeddings.py
    ```
    This will generate the train and test embeddings in the files `train_embedding.txt` and `test_embedding.txt` respectively.
6. To evaluate the model run:
    ```bash
    python eval.py
    ```

7. If you wish to change the architecture, change the hyperparameters or understand the implementation further see the [report file](image_sim_using_deep_ranking.pdf)

## Presentation & Report
* [Presentation](https://docs.google.com/presentation/d/1xaKeIYj5TqKzvNuD_WDcW9UHhT6Qf2lQaFQRUTULKuM/edit?usp=sharing)
* [Report](https://drive.google.com/file/d/1DW1zgqqkWmUGaa6l_QywU1uA5-DIMTRu/view)
## Further Resources:
* [Deep Ranking paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42945.pdf)
* [Project proposal](https://docs.google.com/document/d/1E-2L40X_JUdAb0NssXYnlJTNekLNMuN0z_-Z9KvrexQ/edit)
