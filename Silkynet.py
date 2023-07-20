# silkynet.py
# An implementation of U-Net refered from https://arxiv.org/abs/1505.04597
# based on keras with tensorflow backend.
#

import os 
import numpy as np
import keras.backend
import matplotlib.pyplot as plt
from json import dump, load
from keras.optimizers import adam_v2
from keras.models import Model, load_model
from keras.layers import Conv2D, Input, MaxPooling2D, UpSampling2D, Concatenate
from keras.preprocessing.image import load_img, array_to_img, img_to_array
from keras.callbacks import ModelCheckpoint


class Unet(object):
    """

    1. Train network on silkworm training dataset where the images and masks are seperated in different directories;
    2. Predict masks on test dataset where the images and masks are seperated in different directories;
    3. Save the model file and training history for future use.
    """

    def __init__(self, img_rows=512, img_cols=512, batch_size=4): # Specify the input image size here
        self.img_scale = (img_rows, img_cols, 3)
        self.batch_size = batch_size
        self.train_size = None
        self.valid_size = None
        self.test_size = None


    def _generate_data(self, image_dir, label_dir=None):
        imgnames = sorted(os.listdir(image_dir))
        labelnames = [] if label_dir is None else sorted(os.listdir(label_dir))
        
        while True:
            #random_indices = np.random.choice(np.arange(len(imgnames)), self.batch_size)
            for start in range(0, len(imgnames), self.batch_size):
                data = []
                labels = []
                end = min(start+self.batch_size, len(imgnames))
                #for i in random_indices:
                for i in range(start, end): 
                    image_path = os.path.join(image_dir, imgnames[i])
                    image = load_img(image_path, target_size=self.img_scale)
                    image_arr = img_to_array(image)/255
                    data.append(image_arr)
                    
                    if label_dir is not None:
                        label_path = os.path.join(label_dir, labelnames[i])
                        mask = load_img(label_path, target_size=self.img_scale, color_mode="grayscale")
                        mask_arr = img_to_array(mask)/np.max(mask)
                        labels.append(mask_arr[:, :, 0])
                        
                data = np.array(data)
                if len(labels) == 0:
                    yield data
                else:
                    labels = np.array(labels)
                    yield data, labels.reshape(-1, self.img_scale[0], self.img_scale[1], 1)

    def _generate_test_data(self, image_dir,start=None,end=None):
        imgnames = sorted(os.listdir(image_dir))
        data = []
        for i in range(start,end): 
            image_path = os.path.join(image_dir, imgnames[i])
            image = load_img(image_path, target_size=self.img_scale)
            image_arr = img_to_array(image)/255
            data.append(image_arr)
        data = np.array(data)
        yield data

    def load_data(self, train_image_dir, train_label_dir, valid_image_dir, valid_label_dir, test_image_dir):
        self.train_size = len(os.listdir(train_image_dir))
        self.valid_size = len(os.listdir(valid_image_dir))
        self.test_size = len(os.listdir(test_image_dir))
        
        train_generator = self._generate_data(train_image_dir, train_label_dir)
        valid_generator = self._generate_data(valid_image_dir, valid_label_dir)
        test_generator = self._generate_data(test_image_dir)
        
        return train_generator, valid_generator, test_generator

    def load_test_data(self, test_image_dir,start,end):
        self.test_size = len(os.listdir(test_image_dir))
        test_generator = self._generate_test_data(test_image_dir,start,end)        
        return test_generator


    def _down(self, input_layer, filters, pool=True):
        conv1 = Conv2D(filters, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
        residual = Conv2D(filters, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
        if pool:
            max_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(residual)
            return max_pool, residual
            
        return residual


    def _up(self, input_layer, residual, filters):
        filters=int(filters)
        upsample = UpSampling2D(size=(2, 2))(input_layer)
        upconv = Conv2D(filters, kernel_size=(2, 2), padding="same")(upsample)
        concat = Concatenate(axis=3)([residual, upconv])
        conv1 = Conv2D(filters, kernel_size=(3, 3), padding='same', activation='relu')(concat)
        conv2 = Conv2D(filters, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
        
        return conv2


    def dice_coef(self, y_true, y_pred):
        smooth = 1e-4
        y_true_f = keras.backend.flatten(y_true)
        y_pred_f = keras.backend.flatten(y_pred)
        intersection = keras.backend.sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (keras.backend.sum(y_true_f) + keras.backend.sum(y_pred_f) + smooth)
        
        return score


    def model(self):

        # refered from https://www.kaggle.com/ecobill/u-nets-with-keras
        filters = 64
        input_layer = Input(shape=self.img_scale)
        layers = [input_layer]
        residuals = []
        
        down1, res1 = self._down(input_layer, filters)
        residuals.append(res1)
        filters *= 2
        down2, res2 = self._down(down1, filters)
        residuals.append(res2)
        filters *= 2
        down3, res3 = self._down(down2, filters)
        residuals.append(res3)
        filters *= 2
        down4, res4 = self._down(down3, filters)
        residuals.append(res4)
        filters *= 2
        down5 = self._down(down4, filters, pool=False)

        up1 = self._up(down5, residual=residuals[-1], filters=filters/2)
        filters /= 2
        up2 = self._up(up1, residual=residuals[-2], filters=filters/2)
        filters /= 2
        up3 = self._up(up2, residual=residuals[-3], filters=filters/2)
        filters /= 2
        up4 = self._up(up3, residual=residuals[-4], filters=filters/2)
        
        output_layer = Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid")(up4)
        
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer = adam_v2.Adam(lr=1e-4), loss = 'binary_crossentropy', metrics = [self.dice_coef])
        model.summary()

        return model


    def train(self, train_generator, validation_generator, n_epoch=20, train_steps=8, save_history=False,useModel=False,modelPath=''):
        if useModel is False:
            print('Making an empty model')
            model = self.model()
        else:
            print('Loading the specified model')
            model = load_model(modelPath, custom_objects=dict(dice_coef=self.dice_coef))
        model_checkpoint = ModelCheckpoint('Unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
        train_info = model.fit_generator(train_generator, steps_per_epoch=train_steps, epochs=n_epoch, verbose=1, validation_data=validation_generator, validation_steps=2, callbacks=[model_checkpoint])
        #train_info = model.fit_generator(train_generator, steps_per_epoch=self.train_size//self.batch_size, epochs=n_epoch, verbose=1, validation_data=validation_generator, validation_steps=self.valid_size//self.batch_size, callbacks=[model_checkpoint])
        if save_history is True:
            with open('train_history_unet.json', mode='w', encoding='utf-8') as json_file:
                dump(train_info.history, json_file)


    def plot_history(self, n_epoch):
        with open('train_history_unet.json', mode='r', encoding='utf-8') as json_file:
            history = load(json_file)
        
        plt.style.use('ggplot')
        plt.figure()
        plt.plot(np.arange(0, n_epoch), history['loss'], label='train loss')
        plt.plot(np.arange(0, n_epoch), history['val_loss'], label='val_loss')
        plt.plot(np.arange(0, n_epoch), history['dice_coef'], label='train_dice_coef')
        plt.plot(np.arange(0, n_epoch), history['val_dice_coef'], label='val_dice_coef')
        plt.title('Training performance of U-net')
        plt.xlabel('#Epochs')
        plt.ylabel('Loss/Dice_coef')
        plt.legend(loc='best')
        plt.savefig('loss_dicecoef_unet.png')


    def predict(self, weight_dir, test_generator, test_label_dir,test_image_dir,start=None):
        imgnames = sorted(os.listdir(test_image_dir))
        model = load_model(weight_dir, custom_objects=dict(dice_coef=self.dice_coef))
        masks_arr_predicted = model.predict_generator(test_generator, steps=1)
        
        for i in range(masks_arr_predicted.shape[0]):
            mask_arr_predicted = masks_arr_predicted[i]*255
#            print(mask_arr_predicted[i][0:50])
            mask_predicted = array_to_img(mask_arr_predicted)
#            plt.imshow(mask_predicted)
#            plt.show()
            test_label_path = os.path.join(test_label_dir, '{}_unet.jpeg'.format(imgnames[i+start]))
            mask_predicted.save(test_label_path)
            


import sys, getopt

def main(argv):
    modelPath = ''
    dataDir = ''
    predict = 1
    useModel = False
    dataPath = False
    try:
        opts, args = getopt.getopt(argv,"htm:d:",["mfile=","dfile="])
    except getopt.GetoptError:
        print ('For training and prediction unet.py -t -m <modelpath> -d <data-directory>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('For training and prediction unet.py -t -m <modelpath> -d <data-directory>')
            print ('For prediction only unet.py -m <modelpath> -d <data-directory>')
            print (' ModelPath and Data directory are optional, Default data directory is $script_location/data')
            sys.exit()
        elif opt in ("-t"):
            predict = 0
        elif opt in ("-m", "--mfile"):
            useModel = True
            modelPath = arg
        elif opt in ("-d", "--dfile"):
            dataPath=True
            dataDir = arg
    if dataPath is False:
        data_dir = os.path.join('.', 'data')
    else:
        data_dir = os.path.join(dataDir, 'data')
        
    train_image_dir = os.path.join(data_dir, 'larvaTrain', 'img')
    train_label_dir= os.path.join(data_dir, 'larvaTrain', 'label')
    valid_image_dir = os.path.join(data_dir, 'larvaTrain', 'validation_img')
    valid_label_dir = os.path.join(data_dir, 'larvaTrain', 'validation_label')
    test_image_dir = os.path.join(data_dir, 'larvaTest', 'img')
    test_label_dir = os.path.join(data_dir, 'larvaTest', 'label')

    n_epoch = 30
    unet = Unet()

    if predict == 0: 
        if useModel is True:  # Use a previously trained model
            train_gen, valid_gen, test_gen = unet.load_data(train_image_dir, train_label_dir, valid_image_dir, valid_label_dir, test_image_dir)
            unet.train(train_gen, valid_gen, n_epoch=n_epoch, save_history=True,useModel=True,modelPath=modelPath)
        else:
            unet.train(train_gen, valid_gen, n_epoch=n_epoch, save_history=True,useModel=False,modelPath=modelPath)
        unet.plot_history(n_epoch=n_epoch)
        unet.predict(modelPath, test_gen, test_label_dir)
    else:
        for i in range(0,68,4):
            test_gen = unet.load_test_data(test_image_dir,start=i,end=i+4)
            unet.predict(modelPath, test_gen, test_label_dir,test_image_dir,start=i)

if __name__ == "__main__":
   main(sys.argv[1:])




