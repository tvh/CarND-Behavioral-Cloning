import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model, Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Input, merge
import argparse
import os
import cv2
import matplotlib.image as mpimg

np.random.seed(0)

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 160, 320, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

def load_image(image_file):
    """
    Load RGB images from a file
    """
    return mpimg.imread(image_file)

def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)

def rgb2yuv(image):
    """
    Convert the image from RGB to YUV like NVIDIA
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

def choose_image(center, left, right, steering_angle, speed):
    """
    Randomly choose an image from the center, left or right, and adjust
    the steering angle.
    """
    choice = np.random.choice(3)

    if (choice == 1):
        new_steering_angle = steering_angle + 0.3
        return load_image(left), steering_angle + 0.3, speed*(max(1, 1 - abs(new_steering_angle) + abs(steering_angle)))
    elif (choice == 2):
        new_steering_angle = steering_angle - 0.3
        return load_image(right), new_steering_angle, speed*(max(1, 1 - abs(new_steering_angle) + abs(steering_angle)))
    return load_image(center), steering_angle, speed

def random_flip(image, steering_angle):
    choice = np.random.choice(2)
    if choice == 0:
        return cv2.flip(image,1), -steering_angle
    return image, steering_angle

def augment(center, left, right, steering_angle, speed):
    """
    Generate an augmented image with associated steering commands
    """
    image, steering_angle, speed = choose_image(center, left, right, steering_angle, speed)
    image, steering_angle = random_flip(image, steering_angle)
    return image, steering_angle, speed

def preprocess(image):
    """
    Combine all preprocess functions into one
    """
#    image = resize(image)
    image = rgb2yuv(image)
    return image

def load_data(args):
    """
    Load training data and split it into training and validation set
    """
    data_df = pd.read_csv(os.path.join(args.data_dir, 'driving_log.csv'))

    X = data_df[['center', 'left', 'right']].values
    y = data_df[['steering','speed']].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)

    return X_train, X_valid, y_train, y_valid

def batch_generator(image_paths, all_outputs, batch_size, augment_data=False, output_speed=False):
    """
    Generate batches for training/validation
    """
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    outputs = np.empty([batch_size, 1])
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            steering_angle, speed = all_outputs[index]
            if (augment_data):
                image, steering_angle, speed = augment(center, left, right, steering_angle, speed)
            else:
                image = load_image(center)
            # add the image and steering angle to the batch
            images[i] = preprocess(image)
            if (output_speed):
                outputs[i] = speed
            else:
                outputs[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, outputs

def build_models(args):
    """
    NVIDIA model: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
    """
    inputs = Input(shape=INPUT_SHAPE)
    x = Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE)(inputs)
    x = Conv2D(24, 5, 5, activation='elu', subsample=(2, 2))(x)
    x = Conv2D(36, 5, 5, activation='elu', subsample=(2, 2))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(48, 5, 5, activation='elu', subsample=(2, 2))(x)
    x = Conv2D(64, 3, 3, activation='elu')(x)
    x = Conv2D(64, 3, 3, activation='elu')(x)
    x = Flatten()(x)

    x1 = Dense(100, activation='elu')(x)
    x1 = Dense(50, activation='elu')(x1)
    x1 = Dense(10, activation='elu')(x1)
    x1 = Dense(1)(x1)
    steering_model = Model(inputs, x1)

    x2 = Dense(50, activation='elu')(x)
    x2 = Dense(10, activation='elu')(x2)
    x2 = Dense(1)(x2)
    x2 = Lambda(lambda n: n*30)(x2)
    speed_model = Model(inputs, x2)

    merged = merge([x1, x2], mode='concat', concat_axis=1)
    merged_model = Model(inputs, merged)
    merged_model.summary()

    return steering_model, speed_model, merged_model

class ModelCheckpointOverrideModel(ModelCheckpoint):
    def __init__(
            self, model, filepath, monitor='val_loss', verbose=0,
            save_best_only=False, save_weights_only=False,
            mode='auto', period=1):
        super(ModelCheckpointOverrideModel, self).__init__(
            filepath, monitor=monitor, verbose=verbose,
            save_best_only=save_best_only, save_weights_only=save_weights_only,
            mode=mode, period=period)

        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        self.modelToSave = model

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.modelToSave.save_weights(filepath, overwrite=True)
                        else:
                            self.modelToSave.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch, filepath))
                if self.save_weights_only:
                    self.modelToSave.save_weights(filepath, overwrite=True)
                else:
                    self.modelToSave.save(filepath, overwrite=True)

def train_model(steering_model, speed_model, full_model, args, X_train, X_valid, y_train, y_valid):
    """
    Train the model
    This trains the steering first and then the speed.
    I don't actually care all that much about the speed, so this is fine.
    """
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1, mode='auto')

    # Steering model
    if (args.steering_model):
        print("Loading old model for steering...")
        old_steering_model = load_model(args.steering_model)
        old_config = old_steering_model.get_config()[0]
        old_first_layer_class = old_config['class_name']
        if (old_first_layer_class == 'Input'):
            old_layers = old_steering_model.layers[1:]
        else:
            old_layers = old_steering_model.layers
        for i in range(len(old_steering_model.layers)):
            steering_model.layers[i+1].set_weights(old_layers[i].get_weights())
    else:
        # Save a snapshot after each epoch
        checkpoint_steering = ModelCheckpoint(
            'model-steering-{epoch:03d}.h5',
            monitor='val_loss',
            verbose=1,
            save_best_only=args.save_best_only,
            mode='auto'
        )

        steering_model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))

        steering_model.fit_generator(
            batch_generator(X_train, y_train, args.batch_size, augment_data=True),
            args.samples_per_epoch,
            args.nb_epoch,
            max_q_size=1,
            validation_data=batch_generator(X_valid, y_valid, args.batch_size),
            nb_val_samples=len(X_valid),
            callbacks=[checkpoint_steering, earlyStopping],
            verbose=1
        )

    # Freeze all the layers involved in steering
    for layer in steering_model.layers:
        layer.trainable = False

    # Save a snapshot after each epoch
    checkpoint = ModelCheckpointOverrideModel(
        full_model,
        'model-{epoch:03d}.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=args.save_best_only,
        mode='auto'
    )

    speed_model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))

    speed_model.fit_generator(
        batch_generator(X_train, y_train, args.batch_size, augment_data=True, output_speed=True),
        args.samples_per_epoch,
        args.nb_epoch,
        max_q_size=1,
        validation_data=batch_generator(X_valid, y_valid, args.batch_size, output_speed=True),
        nb_val_samples=len(X_valid),
        callbacks=[checkpoint, earlyStopping],
        verbose=1
    )

def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    if (s == 'true' or s == 'yes' or s == 'y' or s == '1'):
        return True
    elif (s == 'false' or s == 'no' or s == 'n' or s == '0'):
        return False
    raise ValueError

def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str)
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=20)
    parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=20000)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=400)
    parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2b,   default='true')
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
    parser.add_argument('-p', help='steering model',        dest='steering_model',    type=str)
    args = parser.parse_args()

    #print parameters
    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    #load data
    data = load_data(args)
    #build model
    steering_model, speed_model, full_model = build_models(args)
    #train model on data, it saves as model.h5
    train_model(steering_model, speed_model, full_model, args, *data)


if __name__ == '__main__':
    main()
