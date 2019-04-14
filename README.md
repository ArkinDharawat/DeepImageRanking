# DeepImageRanking
Image Similarity using Deep Ranking


## Resources:
* [Deep Ranking paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42945.pdf)
* [398 Project Description](https://courses.engr.illinois.edu/ie534/fa2018/ImageRankingProject.pdf)
* [Project proposal](https://docs.google.com/document/d/1E-2L40X_JUdAb0NssXYnlJTNekLNMuN0z_-Z9KvrexQ/edit)
* [Implementation in Keras](https://github.com/akarshzingade/image-similarity-deep-ranking)

## Documentation
1. Download the TinyImageNet dataset
2. Change the folder structure of the validation set by running:
    ```python
    python transform_validation.py
    ```
3. Now to sample triplets we run:
    ```python
     python triplet_sampler.py --training 1 --num_pos_images 1 --num_neg_images 1
    ```
    Our command-line arugments are as follows:
      * training: 0 or 1 to sample from train or validation set
      * num_pos_images: the number of positive images per query image
      * num_neg_images: the number of negative images per query image