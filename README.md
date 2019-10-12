# TensorFlow-fine-tune-VGG16-Alexnet-models
In this project, finetune process is clearly demonstrated with TensorFlow. 
### Download the pre-trained models
alexnet.npy and vgg16.npy.

Download link: https://drive.google.com/drive/folders/1nDvd3HwPIRlPTn8UJBT7jLNuiwPnisWi?usp=sharing
### Generate image list for training and testing
Generate image list (.txt) from a folder, with format 'path/to/img/ label', each sub-folder contains all images belong to one subject. The name of the sub-foler is the label, start from 0 to N.
```
python3 generate_imglist
```

Generate train.txt and test.txt from image_list.txt
```
python3 generate_train_test_list
```
### Parameters
Choose model with Line 50 in 'finetune.py'. To fine-tune other models, add the defination of model to 'model.py'.
  ```
  pred = Model.vgg16(x, keep_var, n_classes) # Model.alexnet(x, keep_var, n_classes) 
  ```
Training parameters in 'finetune.py'
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
To select weights to be restored from the pre-trained model, modify Line 90 in 'finetune.py'
   ```
   load_with_skip(weight_file, sess, ['fc8'])  # Skip weights from fc8
   ```
To select which layer to be fine-tuned, modify 'model.py' with trainable=True/False.
### Start the finetune process
```
python3 finetune.py train.txt test.txt vgg16.npy
```
