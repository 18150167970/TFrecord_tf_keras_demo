# TFrecord_tf_keras_demo #

This work do:

1. Convert image dataset to TFrecord. 
2. Using TFrecord to train tf.keras model, and save model.

How to use:

please modify data path of code with you use. 

data format

```
--data    
    --trainset
        --classname1
            --file1
            --file2
            ....
        --classname2
```

2020.5.7 update:

add data augmentation, if you want use it, please modify Images_to_TFrecords.py, set is_augmentation True.

[blog](https://blog.csdn.net/a362682954/article/details/105960320)   

