import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model, Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Input
import argparse
import os
import cv2
import matplotlib.image as mpimg

np.random.seed(0)

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

# This should be close to the maximum speed
SPEED_SCALING_FACTOR = 30

def load_image(image_file):
    """
    Load RGB images from a file
    """
    return mpimg.imread(image_file.strip())

def crop(image):
    """
    Crop the image to the final dimension, keeping the center.
    """
    height, width = image.shape[:2]
    y_start = int((height - IMAGE_HEIGHT) / 2)
    x_start = int((width - IMAGE_WIDTH) / 2)
    return image[y_start:y_start+IMAGE_HEIGHT, x_start:x_start+IMAGE_WIDTH]

def rgb2yuv(image):
    """
    Convert the image from RGB to YUV.
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

def modify_speed(old_steering_angle, new_steering_angle, speed):
    """
    Adjust the speed for a modified steering angle.
    This is in a sense also a measure how close the action matches the path given byt the training set.
    """
    return speed*(1 - 2*(abs(new_steering_angle) - abs(old_steering_angle)))

def choose_image(center, left, right, steering_angle, speed):
    """
    Randomly choose an image from the center, left or right, and adjust
    the steering angle.
    """
    choice = np.random.choice(3)

    if (choice == 1):
        new_steering_angle = steering_angle + 0.2
        return load_image(left), new_steering_angle, modify_speed(steering_angle, new_steering_angle, speed)
    elif (choice == 2):
        new_steering_angle = steering_angle - 0.2
        return load_image(right), new_steering_angle, modify_speed(steering_angle, new_steering_angle, speed)
    return load_image(center), steering_angle, speed

def random_flip(image, steering_angle):
    """
    Randomly flip the image ans steering angle.
    """
    choice = np.random.choice(2)
    if choice == 0:
        return cv2.flip(image,1), -steering_angle
    return image, steering_angle

def random_translate(image, steering_angle, speed, range_x=60, range_y=20):
    """
    Randomly translate the image along the x and y axis.
    We want data closer to the reality to dominate.
    I use a triangle distribution, as it is bounded and is less likely to produce larger values.
    """
    trans_x = np.random.triangular(left=-range_x, right=range_x, mode=0)
    trans_y = np.random.triangular(left=-range_y, right=range_y, mode=0)
    new_steering_angle = steering_angle + trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, new_steering_angle, modify_speed(steering_angle, new_steering_angle, speed)

def augment(center, left, right, steering_angle, speed):
    """
    Generate an augmented image with associated steering commands
    """
    image, steering_angle, speed = choose_image(center, left, right, steering_angle, speed)
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle, speed = random_translate(image, steering_angle, speed)
    return image, steering_angle, speed

def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    image = crop(image)
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

def batch_generator(image_paths, all_outputs, batch_size, augment_data=False):
    """
    Generate batches for training/validation
    """
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steering_outputs = np.empty([batch_size, 1])
    speed_outputs = np.empty([batch_size, 1])
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            steering_angle, speed = all_outputs[index]
            # Only yield cases for going straight about 20% of the time. This should combat the bias
            if (abs(steering_angle) > 0.05) or (np.random.rand() < 0.2):
                if (augment_data):
                    image, steering_angle, speed = augment(center, left, right, steering_angle, speed)
                else:
                    image = load_image(center)
                images[i] = preprocess(image)
                steering_outputs[i] = steering_angle
                speed_outputs[i] = speed/SPEED_SCALING_FACTOR
                i += 1
                if i == batch_size:
                    break
        yield images, [steering_outputs, speed_outputs]

def build_model(args):
    """
    NVIDIA model: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
    """
    inputs = Input(shape=INPUT_SHAPE)
    x = Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE)(inputs)
    x = Conv2D(24, 5, 5, activation='elu', subsample=(2, 2))(x)
    x = Conv2D(36, 5, 5, activation='elu', subsample=(2, 2))(x)
    x = Conv2D(48, 5, 5, activation='elu', subsample=(2, 2))(x)
    x = Conv2D(64, 3, 3, activation='elu')(x)
    x = Conv2D(64, 3, 3, activation='elu')(x)
    x = Flatten()(x)

    x1 = Dense(100, activation='elu', name='steering_1')(x)
    x1 = Dense(50, activation='elu', name='steering_2')(x1)
    x1 = Dense(10, activation='elu', name='steering_3')(x1)
    steering_output = Dense(1, name='steering_output')(x1)

    x2 = Dense(30, activation='elu', name='speed_1')(x)
    x2 = Dense(10, activation='elu', name='speed_2')(x2)
    speed_output = Dense(1, name='speed_output')(x2)

    model = Model(inputs, [steering_output, speed_output])
    model.summary()

    return model

def train_model(model, args, X_train, X_valid, y_train, y_valid):
    """
    Train the model
    This trains the steering first and then the speed.
    I don't actually care all that much about the speed, so this is fine.
    """
    earlyStopping = EarlyStopping(monitor='val_steering_output_loss', min_delta=0, patience=4, verbose=1, mode='auto')

    # Save a snapshot after each epoch
    checkpoint = ModelCheckpoint(
        'model-{epoch:03d}.h5',
        monitor='val_steering_output_loss',
        verbose=1,
        save_best_only=args.save_best_only,
        mode='auto'
    )

    # Compile the model. I don't care too much about the speed, so git it a smaller loss_weight.
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate), loss_weights=[1., 0.1])

    model.fit_generator(
        batch_generator(X_train, y_train, args.batch_size, augment_data=True),
        args.samples_per_epoch,
        args.nb_epoch,
        max_q_size=1,
        validation_data=batch_generator(X_valid, y_valid, args.batch_size),
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
