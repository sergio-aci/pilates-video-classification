from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def model_initiation():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  ## Not trainable weights
    model = Sequential([base_model])
    model.add(Flatten())
    model.add(Dense(1024, activation='relu', input_shape=(25088,)))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    return model

def create_model(optimizer='adam'):
    model = model_initiation()
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model

def hypertuning(X_train, y_train, groups):
    print('Begin setting up')
    model = KerasClassifier(build_fn=create_model, verbose=2)

    epochs = [10, 20, 40]
    batch_size = [8, 16, 32]
    optimizer = ['rmsprop','adam']

    param_distributions = dict(epochs=epochs, batch_size=batch_size,
                               optimizer=optimizer,
                            #   init_mode=init_mode,
                              )

    N_SPLITS = 5

    print('Cross Validating...')
    rscv = RandomizedSearchCV(estimator=model,
                              param_distributions=param_distributions,
                              n_iter=10,
                              cv=StratifiedGroupKFold(n_splits=N_SPLITS,
                                                      shuffle=True,
                                                      random_state=28),
                              random_state=28)
    rscv.fit(X=X_train, y=y_train, groups=groups)
    print(rscv.cv_results_)
    print()
    print(rscv.best_score_)
    print()
    print(rscv.best_params_)
    print()
    print(rscv.best_index_)
    print()
    print(rscv.scorer_)
    print('End setting up')
    return rscv.best_params_

def train(X_train, y_train, best_params):

    print('Begin training with the best parameters')
    mcp_save = ModelCheckpoint('weight.hdf55', save_best_only=True, monitor='accuracy')
    es = EarlyStopping( monitor='accuracy', patience=5, restore_best_weights=True)
    tb = TensorBoard(log_dir='logs')
    model = KerasClassifier(build_fn=create_model, verbose=2,
                            optimizer=best_params['optimizer'])
    model.fit(x=X_train, y=y_train,
              epochs=best_params['epochs']+3,
              batch_size=best_params['batch_size'],
              callbacks=[mcp_save, es, tb])
    print('End training')
    return