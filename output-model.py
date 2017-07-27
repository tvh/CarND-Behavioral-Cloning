import argparse

from keras.utils import plot_model
from keras.models import load_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Model as Graph')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image',
        type=str,
        help='Path to output image'
    )
    args = parser.parse_args()

    model = load_model(args.model)
    model.summary()

    plot_model(model, to_file=args.image, show_shapes=True)
