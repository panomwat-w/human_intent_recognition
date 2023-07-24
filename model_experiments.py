import os
import json
import time
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import wandb

fname_list = [
    "win_110723.json",
    "ken_110723.json",
    "eye_110723.json",
    "boom_110723.json",
    "ploy_120723.json",
    "ken_120723.json",
    "ming_120723.json",
    "belle_120723.json",
    "belle_left_120723.json",
]

root_path = "dataset"

# Read and combine json files
def combine_data(fname_list, root_path):
    fname_list = [os.path.join(root_path, f) for f in fname_list]
    data = []
    for fname in fname_list:
        data += json.load(open(fname))
    
    return data

def find_max_len_and_class(data):
    max_len = max([len(x['motion']) for x in data]) + 4 # find max length of the sequence to perform zero padding plus a buffer
    num_classes = max(x['label'] for x in data) + 1 # find the total number of classes

    return max_len, num_classes

def data_augment(tmp_x, n=5, translation=True, scale=True, origin=(0.04, 0.00, 0.02), scale_parameters=(1.0, 0.2), translation_parameters=(0.0, 0.03)):
    augment_data = []
    tmp_x_aug = tmp_x.copy()

    for i in range(n):

        if scale:
            mean_scale = scale_parameters[0]
            std_scale = scale_parameters[1]
            factor = np.random.normal(loc=mean_scale, scale=std_scale, size=1)
            tmp_x_aug = factor * (tmp_x_aug - np.array(origin)) + np.array(origin)

        if translation:
            mean_translation = translation_parameters[0]
            std_translation = translation_parameters[1]
            offset = np.random.normal(loc=mean_translation, scale=std_translation, size=3)
            tmp_x_aug = tmp_x_aug + offset
            
        augment_data.append(tmp_x_aug)
    augment_data = np.stack(augment_data)
    return augment_data

def data_preparation(data, max_len, num_classes, augment, n_augment, translation, scale, scale_parameters, translation_parameters):
    y = []
    X = []
    X_augment = []
    y_augment = []
    for x in data:
        # One-hot encoder
        tmp_y = np.zeros((num_classes))
        tmp_y[x['label']] = 1.0
        
        tmp_x = np.array(x['motion'])[:,:3] # include only position x,y,z
        augment_data = data_augment(tmp_x, n=n_augment, translation=translation, scale=scale, origin=(0.04, 0.00, 0.02), scale_parameters=scale_parameters, translation_parameters=translation_parameters)
        augment_data = np.concatenate([augment_data, np.zeros((augment_data.shape[0], max_len - augment_data.shape[1], augment_data.shape[2]))], axis=1)
        tmp_x = np.concatenate([tmp_x, np.zeros((max_len - tmp_x.shape[0], tmp_x.shape[1]))]) # zero padding
        y.append(tmp_y)
        X.append(tmp_x)
        y_augment.extend([tmp_y] * (n_augment))
        X_augment.extend(list(augment_data))
    X = np.stack(X)
    y = np.stack(y)
    X_augment = np.stack(X_augment)
    y_augment = np.stack(y_augment)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if augment:
        X_train = np.concatenate([X_train, X_augment])
        y_train = np.concatenate([y_train, y_augment])

    return X_train, X_test, y_train, y_test

def make_model(n_features, num_classes, dropout=0.25, lstm_units=[100], fc_units=[100], bidirectional=True, activation="relu"):
    
    model = tf.keras.Sequential()
    if bidirectional:
        if len(lstm_units) > 1:
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units[0], return_sequences=True), input_shape=(None, n_features)))
            for i in range(1, len(lstm_units)):
                model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units[i])))
        else:
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units[0]), input_shape=(None, n_features)))
    else:
        if len(lstm_units) > 1:
            model.add(tf.keras.layers.LSTM(lstm_units[0], input_shape=(None, n_features), return_sequences=True))
            for i in range(1, len(lstm_units)):
                model.add(tf.keras.layers.LSTM(lstm_units[i]))
        else:
            model.add(tf.keras.layers.LSTM(lstm_units[0], input_shape=(None, n_features)))
    model.add(tf.keras.layers.Dropout(dropout))
    for j in range(len(fc_units)):
        model.add(tf.keras.layers.Dense(fc_units[j], activation=activation))
        model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    
    return model

default_config = {
    "augment": True,
    "n_augment": 5,
    "translation": True,
    "translation_parameters": (0.0, 0.03),
    "scale": True,
    "scale_parameters": (1.0, 0.2),
    "max_len": 170,
    "dropout": 0.25,
    "num_classes": 5,
    "lstm_units": [100],
    "fc_units": [100],
    "bidirectional": True,
    "activation": "relu",
    "epochs": 200,
    "batch_size": 32,
    "optimizer": "adam",
    "prediction_threshold": 0.6
}

config_experiments = [
    {
        "augment": False,
        "n_augment": 1,
        "translation": False,
        "translation_parameters": None,
        "scale": False,
        "scale_parameters": None,
        "max_len": 170,
        "dropout": 0.25,
        "num_classes": 5,
        "lstm_units": [100],
        "fc_units": [100],
        "bidirectional": False,
        "activation": "relu",
        "epochs": 200,
        "batch_size": 32,
        "optimizer": "adam",
        "prediction_threshold": 0.6
    },
    {
        "augment": False,
        "n_augment": 1,
        "translation": False,
        "translation_parameters": None,
        "scale": False,
        "scale_parameters": None,
        "max_len": 170,
        "dropout": 0.25,
        "num_classes": 5,
        "lstm_units": [100],
        "fc_units": [100],
        "bidirectional": True,
        "activation": "relu",
        "epochs": 200,
        "batch_size": 32,
        "optimizer": "adam",
        "prediction_threshold": 0.6
    },
    {
        "augment": True,
        "n_augment": 5,
        "translation": True,
        "translation_parameters": (0.0, 0.03),
        "scale": False,
        "scale_parameters": None,
        "max_len": 170,
        "dropout": 0.25,
        "num_classes": 5,
        "lstm_units": [100],
        "fc_units": [100],
        "bidirectional": True,
        "activation": "relu",
        "epochs": 200,
        "batch_size": 32,
        "optimizer": "adam",
        "prediction_threshold": 0.6
    },
    {
        "augment": True,
        "n_augment": 5,
        "translation": False,
        "translation_parameters": None,
        "scale": True,
        "scale_parameters": (1.0, 0.2),
        "max_len": 170,
        "dropout": 0.25,
        "num_classes": 5,
        "lstm_units": [100],
        "fc_units": [100],
        "bidirectional": True,
        "activation": "relu",
        "epochs": 200,
        "batch_size": 32,
        "optimizer": "adam",
        "prediction_threshold": 0.6
    },
    {
        "augment": True,
        "n_augment": 5,
        "translation": True,
        "translation_parameters": (0.0, 0.03),
        "scale": True,
        "scale_parameters": (1.0, 0.2),
        "max_len": 170,
        "dropout": 0.25,
        "num_classes": 5,
        "lstm_units": [100],
        "fc_units": [100],
        "bidirectional": True,
        "activation": "relu",
        "epochs": 200,
        "batch_size": 32,
        "optimizer": "adam",
        "prediction_threshold": 0.6
    },
    {
        "augment": True,
        "n_augment": 3,
        "translation": True,
        "translation_parameters": (0.0, 0.03),
        "scale": True,
        "scale_parameters": (1.0, 0.2),
        "max_len": 170,
        "dropout": 0.25,
        "num_classes": 5,
        "lstm_units": [100],
        "fc_units": [100],
        "bidirectional": True,
        "activation": "relu",
        "epochs": 200,
        "batch_size": 32,
        "optimizer": "adam",
        "prediction_threshold": 0.6
    },
    {
        "augment": True,
        "n_augment": 10,
        "translation": True,
        "translation_parameters": (0.0, 0.03),
        "scale": True,
        "scale_parameters": (1.0, 0.2),
        "max_len": 170,
        "dropout": 0.25,
        "num_classes": 5,
        "lstm_units": [100],
        "fc_units": [100],
        "bidirectional": True,
        "activation": "relu",
        "epochs": 200,
        "batch_size": 32,
        "optimizer": "adam",
        "prediction_threshold": 0.6
    }
]

def main():
    
    data = combine_data(fname_list, root_path)
    max_len, num_classes = find_max_len_and_class(data)

    for exp_config in config_experiments:
        config = default_config.copy()
        config["num_classes"] = num_classes
        config["max_len"] = max_len
        config["augment"] = exp_config["augment"]
        config["n_augment"] = exp_config["n_augment"]
        config["translation"] = exp_config["translation"]
        config["translation_parameters"] = exp_config["translation_parameters"]
        config["scale"] = exp_config["scale"]
        config["scale_parameters"] = exp_config["scale_parameters"]
        config["dropout"] = exp_config["dropout"]
        config["lstm_units"] = exp_config["lstm_units"]
        config["fc_units"] = exp_config["fc_units"]
        config["bidirectional"] = exp_config["bidirectional"]
        config["activation"] = exp_config["activation"]

        X_train, X_test, y_train, y_test = data_preparation(data, max_len, num_classes, config["augment"], config["n_augment"], config["translation"], config["scale"], config["scale_parameters"], config["translation_parameters"])

        model = make_model(n_features=X_train.shape[2], num_classes=num_classes, dropout=config["dropout"], lstm_units=config["lstm_units"], fc_units=config["fc_units"], bidirectional=config["bidirectional"], activation=config["activation"])

        model_name = f'best_lstm_model_{config["n_augment"]}_{config["translation"]}_{config["translation_parameters"]}_{config["scale"]}_{config["scale_parameters"]}_{config["dropout"]}_{config["lstm_units"]}_{config["fc_units"]}_{config["bidirectional"]}_{config["activation"]}.h5'

        wandb.init(project="human_intention_recognition", entity='pw499', config=default_config, name=model_name)
        wandb.config = config

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join("model_experiments", model_name), save_best_only=True, monitor="val_loss"
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
            ),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
            wandb.keras.WandbCallback(),
        ]

        model.compile(
            optimizer=config["optimizer"],
            loss="categorical_crossentropy",
            metrics=["categorical_accuracy"],
        )

        history = model.fit(
            X_train,
            y_train,
            batch_size=config["batch_size"],
            epochs=config["epochs"],
            callbacks=callbacks,
            validation_split=0.2,
            verbose=1,
        )

        # Load Best Model to evaluate
        model = tf.keras.models.load_model(os.path.join("model_experiments", model_name))
        start = time.time()
        test_loss, test_acc = model.evaluate(X_test, y_test)
        end = time.time()
        inference_time = (end - start)*1000/X_test.shape[0] 
        
        threshold = config["prediction_threshold"]
        y_pred = model.predict(X_test)
        cond = y_pred.max(axis=1)>threshold

        y_pred_bool = np.argmax(y_pred[cond], axis=1)
        y_test_bool = np.argmax(y_test[cond], axis=1)
        report_dict = classification_report(y_test_bool, y_pred_bool, output_dict=True)

        wandb.log({"infer_time": inference_time, "test_acc": test_acc, "test_loss": test_loss})
        wandb.finish()


if __name__ == '__main__':
    main()


