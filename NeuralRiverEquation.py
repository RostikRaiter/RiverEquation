import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, GRU, Dropout
from keras._tf_keras.keras.optimizers import Adam
from scipy.interpolate import make_interp_spline


df = pd.read_csv('river_simulation_dataset.csv')


time_steps = df['Time'].unique()
X_train_dict = {}
X_test_dict = {}
y_train_dict = {}
y_test_dict = {}

for t in time_steps:
    
    subset = df[df['Time'] == t]
    X_subset = subset[['Width', 'Velocity', 'Slope', 'Roughness', 'Length', 'Inflow', 'Position']].values
    y_subset = subset['Depth'].values

    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_subset)

   
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_subset, test_size=0.2, random_state=42)

    
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    
    X_train_dict[t] = X_train
    X_test_dict[t] = X_test
    y_train_dict[t] = y_train
    y_test_dict[t] = y_test


predicted_depths_dict = {}
loss_dict = {}

for t in time_steps:
    
    model = Sequential()
    model.add(GRU(128, activation='relu', input_shape=(X_train_dict[t].shape[1], X_train_dict[t].shape[2]), return_sequences=True))
    model.add(Dropout(0.1))
    model.add(GRU(128, activation='relu', return_sequences=True))
    model.add(Dropout(0.1))
    model.add(Dense(1))

    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mse')

    
    history = model.fit(X_train_dict[t], y_train_dict[t], epochs=300, batch_size=64, validation_data=(X_test_dict[t], y_test_dict[t]), verbose=1)

    
    loss_dict[t] = history.history['loss'][-1]

    
    X_full = df[df['Time'] == t][['Width', 'Velocity', 'Slope', 'Roughness', 'Length', 'Inflow', 'Position']].values
    X_full_scaled = scaler.transform(X_full)
    X_full_reshaped = X_full_scaled.reshape((X_full_scaled.shape[0], 1, X_full_scaled.shape[1]))
    predicted_depths = model.predict(X_full_reshaped).flatten()

    predicted_depths_dict[t] = predicted_depths


df['Predicted_Depth'] = np.nan
for t in time_steps:
    subset = df[df['Time'] == t]
    subset['Predicted_Depth'] = predicted_depths_dict[t]
    df.loc[subset.index, 'Predicted_Depth'] = subset['Predicted_Depth']


segments = np.array_split(df['Position'].unique(), 20)  


segment_loss_dict = {t: [] for t in time_steps}

for t in time_steps:
    for segment in segments:
        segment_indices = df['Position'].isin(segment) & (df['Time'] == t)
        y_true_segment = df.loc[segment_indices, 'Depth'].values
        y_pred_segment = df.loc[segment_indices, 'Predicted_Depth'].values

        if len(y_true_segment) > 0 and len(y_pred_segment) > 0:
            mse = np.mean((y_true_segment - y_pred_segment) ** 2)
            segment_loss_dict[t].append(mse)
        else:
            segment_loss_dict[t].append(np.nan)


normalized_segment_loss_dict = {}

for t in time_steps:
    max_loss = np.nanmax(segment_loss_dict[t])
    normalized_segment_loss_dict[t] = [100 * (loss / max_loss) if not np.isnan(loss) else np.nan for loss in segment_loss_dict[t]]


medium_loss_dict = {}
for t in time_steps:
    medium_loss_dict[t] = np.nanmedian(normalized_segment_loss_dict[t])


plt.figure(figsize=(15, 10))

for t in time_steps:
    subset = df[df['Time'] == t]
    x_values = subset['Position'].values
    y_values_original = subset['Depth'].values
    y_values_predicted = subset['Predicted_Depth'].values

    
    valid_indices = ~np.isnan(y_values_predicted)
    x_values = x_values[valid_indices]
    y_values_original = y_values_original[valid_indices]
    y_values_predicted = y_values_predicted[valid_indices]

    if len(x_values) > 0 and len(y_values_predicted) > 0:
        
        y_min = y_values_original.min()
        y_max = y_values_original.max()
        y_values_predicted_normalized = (y_values_predicted - y_values_predicted.min()) / (y_values_predicted.max() - y_values_predicted.min())
        y_values_predicted_scaled = y_values_predicted_normalized * (y_max - y_min) + y_min

        
        x_smooth = np.linspace(x_values.min(), x_values.max(), 300)
        spl_original = make_interp_spline(x_values, y_values_original, k=3) 
        y_smooth_original = spl_original(x_smooth)

        
        spl_predicted = make_interp_spline(x_values, y_values_predicted_scaled, k=3)
        y_smooth_predicted = spl_predicted(x_smooth)

        plt.plot(x_smooth, y_smooth_original, label=f'Оригінальні дані {t:.2f}с', linestyle='--')
        plt.plot(x_smooth, y_smooth_predicted, label=f'Дані з використанням нейронної мережі {t:.2f}с', linestyle='-')

plt.xlabel('Відстань (м)')
plt.ylabel('Глибина (м)')
plt.title('Зміна глибини річки з плином часу використовуючи нові дані')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()


for t in time_steps:
    print(f'Часовий відрізок {t:.2f}с:')
    print(f'  Середня втрата: {medium_loss_dict[t]:.2f}%')