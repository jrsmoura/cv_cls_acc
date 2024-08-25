import argparse
import os
from auxiliary_functions.aux_functions import (
    get_filepaths_and_labels,
    create_dataloader
)


def preprocess(data_dir: str, output_dir, batch_size: int):
    """
    Preprocess the data and create a dataloader
    :param data_dir:
    :param output_dir:
    :param batch_size:
    :return: dataloader
    """
    filepaths, labels = get_filepaths_and_labels(data_dir)
    dataloader = create_dataloader(filepaths, labels, batch_size)

    os.makedirs(output_dir, exist_ok=True)

    return dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess the data')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()
    preprocess(args.data_dir, args.output_dir, args.batch_size)