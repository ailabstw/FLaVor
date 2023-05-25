import argparse
import os
import pickle

import tensorflow as tf

from flavor.cook.utils import SaveInfoJson, SetEvent, WaitEvent

gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


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
        "--epochs-per-round",
        type=int,
        default=1,
        metavar="N",
        help="number of epochs per round (default: 1)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, metavar="LR", help="learning rate (default: 0.001)"
    )
    args = parser.parse_args()

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.reshape(60000, 28, 28) / 255
    x_test = x_test.reshape(10000, 28, 28) / 255

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

    # Round index
    round_idx = 0

    while True:

        # Wait for the server
        WaitEvent("TrainStarted")

        # Load checkpoint sent from the server
        if round_idx != 0 or os.path.exists(os.environ["GLOBAL_MODEL_PATH"]):
            with open(os.environ["GLOBAL_MODEL_PATH"], "rb") as F:
                weights = pickle.load(F)
            model.set_weights(list(weights["state_dict"].values()))

        # Verify the performance of the global model before training
        loss, accuracy = model.evaluate(x_test, y_test)

        # Save information that the server needs to know
        output_dict = {}
        output_dict["metadata"] = {
            "epoch": round_idx,
            "datasetSize": x_train.shape[0],
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

        model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs_per_round)

        # Save checkpoint
        weights = {"state_dict": {str(key): value for key, value in enumerate(model.get_weights())}}
        with open(os.environ["LOCAL_MODEL_PATH"], "wb") as F:
            pickle.dump(weights, F, protocol=pickle.HIGHEST_PROTOCOL)

        # Tell the server that this round of training work has ended.
        SetEvent("TrainFinished")

        round_idx += 1


if __name__ == "__main__":

    main()
