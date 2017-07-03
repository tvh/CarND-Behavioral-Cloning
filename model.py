import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
import argparse
import os
import cv2
import matplotlib.image as mpimg

np.random.seed(0)

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 80, 160, 3
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

def choose_image(center, left, right, steering_angle):
    """
    Randomly choose an image from the center, left or right, and adjust
    the steering angle.
    Use a bias towards the side images if the steering angle is 0.
    """
    if (steering_angle == 0):
        choice = np.random.choice(5)
    else:
        choice = np.random.choice(3)

    if (choice == 1 or choice == 3):
        return load_image(left), steering_angle + 0.3
    elif (choice == 2 or choice == 4):
        return load_image(right), steering_angle - 0.3
    return load_image(center), steering_angle

def random_flip(image, steering_angle):
    choice = np.random.choice(2)
    if choice == 0:
        return cv2.flip(image,1), -steering_angle
    return image, steering_angle

def augment(center, left, right, steering_angle):
    """
    Generate an augmented image with associated steering commands
    """
    image, steering_angle = choose_image(center, left, right, steering_angle)
    image, steering_angle = random_flip(image, steering_angle)
    return image, steering_angle

def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    image = resize(image)
    image = rgb2yuv(image)
    return image

def load_data(args):
    """
    Load training data and split it into training and validation set
    """
    data_df = pd.read_csv(os.path.join(args.data_dir, 'driving_log.csv'))

    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)

    return X_train, X_valid, y_train, y_valid

def batch_generator(image_paths, steering_angles, batch_size, augment_data=False):
    """
    Generate batches for training/validation
    """
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            steering_angle = steering_angles[index]
            image, steering_angle = augment(center, left, right, steering_angle)
            # add the image and steering angle to the batch
            images[i] = preprocess(image)
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, steers

def build_model(args):
    """
    NVIDIA model: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
    """
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()

    return model

def train_model(model, args, X_train, X_valid, y_train, y_valid):
    """
    Train the model
    """
    # Save a snapshot after each epoch
    checkpoint = ModelCheckpoint(
        'model-{epoch:03d}.h5',
        monitor='val_loss',
        verbose=0,
        save_best_only=args.save_best_only,
        mode='auto'
    )

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))

    model.fit_generator(
        batch_generator(X_train, y_train, args.batch_size, augment_data=True),
        args.samples_per_epoch,
        args.nb_epoch,
        max_q_size=1,
        validation_data=batch_generator(X_valid, y_valid, args.batch_size),
        nb_val_samples=len(X_valid),
        callbacks=[checkpoint],
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
    model = build_model(args)
    #train model on data, it saves as model.h5
    train_model(model, args, *data)


if __name__ == '__main__':
    main()
