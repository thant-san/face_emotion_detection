# Face emotion detection using cnn
## Introduction
 This project purposes is to pedict the emotions of the person's face into seven categories using deep learning convolutional neural network.
 The model is trained on FER-2013 dataset which was published on International Conference on Machine Learning (ICML).This dataset consists 
 of 35887 grayscale, 48x48 sized face images with seven emotions - angry, disgusted, fearful, happy, neutral, sad and surprised.
FER-2013 Using CNN
## Model approch
-use haarcascade to detect the face
-extract the face aera
-predict the extracted face using CNN 
## Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 48, 48, 64)        640       
                                                                 
 batch_normalization (BatchN  (None, 48, 48, 64)       256       
 ormalization)                                                   
                                                                 
 activation (Activation)     (None, 48, 48, 64)        0         
                                                                 
 max_pooling2d (MaxPooling2D  (None, 24, 24, 64)       0         
 )                                                               
                                                                 
 dropout (Dropout)           (None, 24, 24, 64)        0         
                                                                 
 conv2d_1 (Conv2D)           (None, 24, 24, 128)       204928    
                                                                 
 batch_normalization_1 (Batc  (None, 24, 24, 128)      512       
 hNormalization)                                                 
                                                                 
 activation_1 (Activation)   (None, 24, 24, 128)       0         
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 12, 12, 128)      0         
 2D)                                                             
                                                                 
 dropout_1 (Dropout)         (None, 12, 12, 128)       0         
                                                                 
 conv2d_2 (Conv2D)           (None, 12, 12, 512)       590336    
                                                                 
 batch_normalization_2 (Batc  (None, 12, 12, 512)      2048      
 hNormalization)                                                 
                                                                 
 activation_2 (Activation)   (None, 12, 12, 512)       0         
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 6, 6, 512)        0         
 2D)                                                             
                                                                 
 dropout_2 (Dropout)         (None, 6, 6, 512)         0         
                                                                 
 conv2d_3 (Conv2D)           (None, 6, 6, 512)         2359808   
                                                                 
 batch_normalization_3 (Batc  (None, 6, 6, 512)        2048      
 hNormalization)                                                 
                                                                 
 activation_3 (Activation)   (None, 6, 6, 512)         0         
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 3, 3, 512)        0         
 2D)                                                             
                                                                 
 dropout_3 (Dropout)         (None, 3, 3, 512)         0         
                                                                 
 flatten (Flatten)           (None, 4608)              0         
                                                                 
 dense (Dense)               (None, 256)               1179904   
                                                                 
 batch_normalization_4 (Batc  (None, 256)              1024      
 hNormalization)                                                 
                                                                 
 activation_4 (Activation)   (None, 256)               0         
                                                                 
 dropout_4 (Dropout)         (None, 256)               0         
                                                                 
 dense_1 (Dense)             (None, 512)               131584    
                                                                 
 batch_normalization_5 (Batc  (None, 512)              2048      
 hNormalization)                                                 
                                                                 
 activation_5 (Activation)   (None, 512)               0         
                                                                 
 dropout_5 (Dropout)         (None, 512)               0         
                                                                 
 dense_2 (Dense)             (None, 7)                 3591      
                                                                 
=================================================================
Total params: 4,478,727
Trainable params: 4,474,759
Non-trainable params: 3,968

### FER-2013 Dataset

-Dataset link:https://www.kaggle.com/datasets/nicolejyt/facialexpressionrecognitio
### Streamlit
-https://thant-san-face-emotion-detection-myapp-9wg7an.streamlit.app/

