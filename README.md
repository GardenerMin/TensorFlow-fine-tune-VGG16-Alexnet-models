# TensorFlow-fine-tune-VGG16-Alexnet-models
In this project, we demonstrate the finetune process in TensorFlow.
### Download the pre-trained model
alexnet.npy or vgg16.npy
### Generate image list for training and testing
Generate image list (.txt) from a folder, with format 'path/to/img/ label', each sub-folder contains all images belong to one subject. The name of the sub-foler is the label, start from 0 to N.
```
python3 generate_imglist
```

Generate train.txt and test.txt from image_list.txt
```
python3 generate_train_test_list
```
### Parameters setting
Modify the parameters in 'finetune.py'
  ```
  # Learning params  
    learning_rate_init = 0.001   
    decay_steps = 10000
    decay_rate = 0.5

    # Train and dispaly params
    training_iters = 60000
    batch_size = 50         
    display_step = 20
    test_step = 1000
    save_step = 1000

    # Network params
    n_classes = 10575
    keep_rate = 0.5  
   ```
### Start the finetune process
```
python3 finetune.py train.txt test.txt vgg16.npy
```
