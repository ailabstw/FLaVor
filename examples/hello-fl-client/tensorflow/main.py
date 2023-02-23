import argparse
import os
import pickle

import tensorflow as tf
import tensorflow_datasets as tfds

from flavor.cook.utils import SaveInfoJson, SetEvent, WaitEvent


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255.0, label


def main():

    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for training (default: 32)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        metavar="N",
        help="number of epochs to train (default: 300)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, metavar="LR", help="learning rate (default: 0.001)"
    )
    args = parser.parse_args()

    (ds_train, ds_test), ds_info = tfds.load(
        "mnist",
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
    ds_train = ds_train.batch(args.batch_size)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(args.batch_size)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    # Tell the server that all preparations for training have been completed.
    SetEvent("TrainInitDone")

    for epoch in range(args.epochs):

        # Wait for the server
        WaitEvent("TrainStarted")

        # Load checkpoint sent from the server
        if epoch != 0 or os.path.exists(os.environ["GLOBAL_MODEL_PATH"]):
            with open(os.environ["GLOBAL_MODEL_PATH"], "rb") as F:
                weights = pickle.load(F)
            model.set_weights(list(weights["state_dict"].values()))

        # Verify the performance of the global model before training
        loss, accuracy = model.evaluate(ds_test)

        # Save information that the server needs to know
        output_dict = {}
        output_dict["metadata"] = {
            "epoch": epoch,
            "datasetSize": ds_info.splits["train"].num_examples,
            "importance": 1.0,
        }
        output_dict["metrics"] = {
            "accuracy": accuracy,
            "basic/confusion_tp": -1,  # If N/A or you don't want to track, fill in -1.
            "basic/confusion_fp": -1,
            "basic/confusion_fn": -1,
            "basic/confusion_tn": -1,
        }
        SaveInfoJson(output_dict)

        model.fit(ds_train, epochs=1)

        # Save checkpoint
        weights = {"state_dict": {str(key): value for key, value in enumerate(model.get_weights())}}
        with open(os.environ["LOCAL_MODEL_PATH"], "wb") as F:
            pickle.dump(weights, F, protocol=pickle.HIGHEST_PROTOCOL)

        # Tell the server that this round of training work has ended.
        SetEvent("TrainFinished")


if __name__ == "__main__":

    main()
