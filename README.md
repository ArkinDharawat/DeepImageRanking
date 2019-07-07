# DeepImageRanking
Image Similarity using Deep Ranking

## Documentation
1. Download the TinyImageNet dataset
2. Change the folder structure of the validation set by running:
    ```bash
    python transform_validation.py
    ```
3. Now to sample triplets we run:
    ```bash
       python triplet_sampler.py --training 1 --num_pos_images 1 --num_neg_images 1
    ```
    Our command-line arugments are as follows:
      * training: 0 or 1 to sample from train or validation set
      * num_pos_images: the number of positive images per query image
      * num_neg_images: the number of negative images per query image
4. To train the network run:
    ```bash
       python train_net.py --epochs 1 --optim sgd
    ```
    Set choice of optimizer and num of epochs in the same file.
5. To generate embeddings:
    ```bash
        python gen_emebeddings.py
    ```
6. Run accuracy_notebook for accuracy and ranked examples
7. If you wish to change the architecture, modify deep_rank_net.py

## Presentation&Report
* [Presentation](https://docs.google.com/presentation/d/1xaKeIYj5TqKzvNuD_WDcW9UHhT6Qf2lQaFQRUTULKuM/edit?usp=sharing)
* [Report](https://drive.google.com/file/d/1DW1zgqqkWmUGaa6l_QywU1uA5-DIMTRu/view)
## Resources:
* [Deep Ranking paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42945.pdf)
* [Project proposal](https://docs.google.com/document/d/1E-2L40X_JUdAb0NssXYnlJTNekLNMuN0z_-Z9KvrexQ/edit)
* [Implementation in Keras](https://github.com/akarshzingade/image-similarity-deep-ranking)
