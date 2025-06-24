##Код для jup-notebook
import tensorflow as tf
import numpy as np
import pandas as pd
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Найдено {len(physical_devices)} GPU")
    except RuntimeError as e:
        print(f"Ошибка при настройке памяти GPU: {e}")
else:
    print("GPU не найдено, используется CPU")
data = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
data_test = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')
print(data.shape)
print(data_test.shape)

data.head()
tester_2 = data
train = data

data_test = data_test.drop(['label'], axis = 1)
data_test.head()

def reduce_mem_usage(dataframe, dataset):    
    print('Reducing memory usage for:', dataset)
    initial_mem_usage = dataframe.memory_usage().sum() / 1024**2
    
    for col in dataframe.columns:
        col_type = dataframe[col].dtype

        c_min = dataframe[col].min()
        c_max = dataframe[col].max()
        if str(col_type)[:3] == 'int':
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                dataframe[col] = dataframe[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                dataframe[col] = dataframe[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                dataframe[col] = dataframe[col].astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                dataframe[col] = dataframe[col].astype(np.int64)
        else:
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                dataframe[col] = dataframe[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                dataframe[col] = dataframe[col].astype(np.float32)
            else:
                dataframe[col] = dataframe[col].astype(np.float64)

    final_mem_usage = dataframe.memory_usage().sum() / 1024**2
    print('--- Memory usage before: {:.2f} MB'.format(initial_mem_usage))
    print('--- Memory usage after: {:.2f} MB'.format(final_mem_usage))
    print('--- Decreased memory usage by {:.1f}%\n'.format(100 * (initial_mem_usage - final_mem_usage) / initial_mem_usage))

    return dataframe

cols_to_drop = [
    'X697', 'X698', 'X699', 'X700', 'X701', 'X702', 'X703', 'X704', 'X705', 'X706', 
    'X707', 'X708', 'X709', 'X710', 'X711', 'X712', 'X713', 'X714', 'X715', 'X716',
    'X717', 'X864', 'X867', 'X869', 'X870', 'X871', 'X872', 'X104', 'X110', 'X116',
    'X122', 'X128', 'X134', 'X140', 'X146', 'X152', 'X158', 'X164', 'X170', 'X176',
    'X182', 'X351', 'X357', 'X363', 'X369', 'X375', 'X381', 'X387', 'X393', 'X399',
    'X405', 'X411', 'X417', 'X423', 'X429'
]

import numpy as np
train = train.drop(columns=cols_to_drop)
test = data_test
test = test.drop(columns=cols_to_drop)

trainer = reduce_mem_usage(train, "train")
tester = reduce_mem_usage(test, "test")

X = trainer
X = X.drop(['label'], axis = 1)
target = data['label']
print(X.shape)
print(test.shape)
print(target.shape)
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.mixed_precision import set_global_policy
from sklearn.model_selection import train_test_split
scaler_X = MinMaxScaler()
X_scaledd = scaler_X.fit_transform(X)  # Форма: (540000, 840)
scaler_target = MinMaxScaler()
target = scaler_target.fit_transform(target.to_numpy().reshape(-1, 1)).flatten() # Форма: (540000,)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaledd, target, test_size=0.2, random_state=42
)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

print(type(X_train))
print(type(y_train))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization

modell = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(840, 1)),
    BatchNormalization(),
    Dropout(0.2),
    Flatten(),
    Dense(256, activation='relu', input_shape=(840,), kernel_regularizer=l2(0.01)),  # Увеличено до 256 нейронов + L2
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dropout(0.1),
    Dense(1, dtype='float32')  # Выходной слой
])

modell.compile(
    optimizer=Adam(learning_rate=0.0005),  # Уменьшен learning_rate
    loss='mae',
    metrics=['mae']
)

modell.summary()


from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
with tf.device('/GPU:0'):
    history = modell.fit(
        X_train, y_train,
        batch_size=128,
        epochs=7,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )


tester_scaled = scaler_X.transform(tester)  # Форма: (n_samples, 840)

print("tester_scaled shape:", tester_scaled.shape)

with tf.device('/GPU:0'):
    predictions = model.predict(tester_scaled, batch_size=128, verbose=1)

predictions_original = scaler_target.inverse_transform(predictions.reshape(-1, 1)).flatten()

print("\nПервые 5 предсказанных значений (обратно нормализованы):", predictions[:5])

predictions_df = pd.DataFrame({
    'id': np.arange(1, len(predictions_original) + 1),
    'prediction': predictions_original
})
predictions_df.to_csv('predictions.csv', index=False)
print("Предсказания сохранены в 'predictions.csv'")
