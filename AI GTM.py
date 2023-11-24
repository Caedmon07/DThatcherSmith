import tkinter as tk
from tkinter import ttk
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def gas_turbine_engine_model(ambient_pressure, ambient_temperature, rpm, fuel_flow, text_input):
    text_value = len(text_input)
    
    #Add logi for text_input
    compressor_outlet = ambient_pressure * (1 + 0.1 * (rpm / 1000))
    combustor_outlet = compressor_outlet + 0.5 * fuel_flow
    turbine_outlet = combustor_outlet - 0.2 * rpm
    nozzle_outlet = turbine_outlet * np.sqrt(ambient_temperature / turbine_outlet)
    
    combined_output = nozzle_outlet + 0.1 * text_value

    return combined_output

def train_neural_network(X_train, y_train, text_train, epochs, batch_size):
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_train)
    text_sequences = tokenizer.texts_to_sequences(text_train)
    padded_text_sequences = pad_sequences(text_sequences)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(25, activation='relu'),
        tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim = 16, input_length=padded_text_sequences.shap[1]),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit([X_train_scaled, padded_text_sequences], y_train_scaled, epochs=epochs, batch_size=batch_size)

    return model, scaler_X, scaler_y, tokenizer

def predict_with_neural_network(model, scaler_X, scaler_y, tokenizer, numeric_input, text_input):
    numeric_input_scaled = scaler_X.transform(np.array([numeric_input]))
    text_sequence = tokenizer.texts_to_sequences([text_input])
    padded_text_sequence = pad_sequences(text_sequence, maxlen=model.layers[0].input_length)
    output_scaled = model.predict([numeric_input_scaled, padded_text_sequence])
    output = scaler_y.inverse_transform(output_scaled)
    return output.flatten()[0]

class GasTurbineGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Hybrid AI Gas Turbine Model")
        self.geometry("500x500")
        self.create_widgets()
    def create_widgets(self):
        self.label_pressure = ttk.Label(self, text="Ambient Pressure")
        self.entry_pressure = ttk.Entry(self)
        self.label_temperature = ttk.Label(self, text="Ambient Temperature")
        self.entry_temperature = ttk.Entry(self)
        self.label_rpm = ttk.Label(self, text="RPM")
        self.entry_rpm = ttk.Entry(self)
        self.label_fuel_flow = ttk.Label(self, text="Fuel Flow")
        self.entry_fuel_flow = ttk.Entry(self)
        self.label_text_input = ttk.Label(self, text="Text Input: ")
        self.entry_text_input = ttk.Entry(self)

        self.label_prediction = ttk.Label(self, text="AI Prediction")

        self.predict_button = ttk.Button(self, text="Predict", command=self.predict)

        self.label_pressure.grid(row=0, column=0, padx=10, pady=5)
        self.entry_pressure.grid(row=0, column=1, padx=10, pady=5)
        self.label_temperature.grid(row=1, column=0, padx=10, pady=5)
        self.entry_temperature.grid(row=1, column=1, padx=10, pady=5)
        self.label_rpm.grid(row=2, column=0, padx=10, pady=5)
        self.entry_rpm.grid(row=2, column=1, padx=10, pady=5)
        self.label_fuel_flow.grid(row=3, column=0, padx=10, pady=5)
        self.entry_fuel_flow.grid(row=3, column=1, padx=10, pady=5)
        self.label_text_input.grid(row=4, column=0, padx=10, pady=5)
        self.entry_text_input.grid(row=4, column=1, padx=10, pady=5)
        self.predict.button.grid(row=5, column=0, columnspan=2, pady=10)
        self.label_prediction.grid(row=6, column=0, columnspan=2, pady=5)
    
    def predict(self):
        try:
            ambient_pressure = float(self.entry_pressure.get())
            ambient_temperature = float(self.entry_temperature.get())
            rpm = float(self.entry_rpm.get())
            fuel_flow = float(self.entry_fuel_flow.get())
            
            text_input = self.entry_text_input.get()
            
            engine_output = gas_turbine_engine_model(ambient_pressure, ambient_temperature, rpm, fuel_flow, text_input)

            X_train = np.array([[ambient_pressure, ambient_temperature, rpm, fuel_flow]])
            y_train = np.array([engine_output])
            text_train = [text_input]

            model, scaler_X, scaler_y, Tokenizer = train_neural_network(X_train, y_train, text_train, epochs=50, batch_size=50)

            numeric_input = [ambient_pressure, ambient_temperature, rpm, fuel_flow]
            prediction = predict_with_neural_network(model, scaler_X, scaler_y, Tokenizer, numeric_input, text_input)

            self.label_prediction.config(text=f"AI Prediction: {prediction:.2f}")
        except ValueError:
            self.label_prediction.config(text="Invalid input. Please enter numeric values.")

if __name__ == "__main__":
    app = GasTurbineGUI()
    app.mainloop()
    

