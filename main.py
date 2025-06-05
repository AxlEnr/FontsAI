import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from palette import selectColor
import sys
from sklearn.model_selection import train_test_split
url = 'https://tinyurl.com/y2qmhfsr'
data = pd.read_csv(url)
X = data[['RED', 'GREEN', 'BLUE']].values / 255  # Normalizar datos
y = data['LIGHT_OR_DARK_FONT_IND'].values

# One-hot encoding
Y = np.zeros((y.size, 2))
Y[np.arange(y.size), y] = 1

# Separar datos
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Párametros de la red neuronal
input_size = 3
hidden_size = 8
output_size = 2

# Inicializar pesos
w1 = np.random.randn(input_size, hidden_size) * 0.1
b1 = np.zeros((1, hidden_size))
w2 = np.random.randn(hidden_size, output_size) * 0.1
b2 = np.zeros((1, output_size))

# Activaciones
def relu(x): return np.maximum(0, x)
def d_relu(x): return (x > 0).astype(float)
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Forward propagation
def forward(X):
    z1 = np.dot(X, w1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

# Backward propagation
def backward(X, Y, z1, a1, z2, a2):
    m = X.shape[0]
    dz2 = (a2 - Y) / m
    dw2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)

    da1 = np.dot(dz2, w2.T)
    dz1 = da1 * d_relu(z1)
    dw1 = np.dot(X.T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)
    
    return dw1, db1, dw2, db2

# Entrenamiento por epocas
lr = 0.05
for epoch in range(1000):
    z1, a1, z2, a2 = forward(X_train)
    dw1, db1, dw2, db2 = backward(X_train, Y_train, z1, a1, z2, a2)

    w1 -= lr * dw1
    b1 -= lr * db1
    w2 -= lr * dw2
    b2 -= lr * db2

# Precisión de la red realizada por back propagation
_, _, _, a2_test = forward(X_test)
pred = np.argmax(a2_test, axis=1)
true = np.argmax(Y_test, axis=1)
accuracy = np.mean(pred == true)
print(f"Precisión en test: {accuracy * 100:.2f}%")

def prediccion_fuente(color_rgb):
    color_np = np.array(color_rgb).reshape(1, -1) / 255
    _, _, _, output = forward(color_np)
    return np.argmax(output, axis=1)

def mostrar_prediccion():
    while True:
        color_rgb = selectColor()
        if not color_rgb:
            break

        color_list = list(map(int, color_rgb))
        color_predicho = prediccion_fuente(color_list)
        color_hex = '#%02x%02x%02x' % tuple(color_list)

        window = tk.Tk()
        window.title("Recomendación de Fuente")
        window.geometry("400x260")
        window.configure(bg="#f4f4f4")
        window.overrideredirect(True)

        frame = tk.Frame(window, bg="white", bd=3, relief="ridge")
        frame.place(relx=0.5, rely=0.5, anchor="center", width=400, height=260)

        titulo = tk.Label(frame, text="Resultado de la Predicción", font=("Segoe UI", 14, "bold"), bg="white")
        titulo.pack(pady=(10, 5))

        texto_recomendado = "🎨 Usa TEXTO CLARO" if color_predicho[0] == 0 else "🎨 Usa TEXTO OSCURO"
        resultado = tk.Label(frame, text=texto_recomendado, font=("Segoe UI", 12), bg="white", fg="#333333")
        resultado.pack(pady=5)

        canvas = tk.Canvas(frame, width=120, height=50, bg=color_hex, bd=1, relief="solid")
        canvas.pack(pady=5)

        def cerrar_y_repetir():
            window.destroy()
        
        def cerrar():
            window.destroy()
            sys.exit() 

        boton_reintentar = tk.Button(
            frame, text="Seleccionar otro color", command=cerrar_y_repetir,
            bg="#2196F3", fg="white", font=("Segoe UI", 10, "bold"), relief="flat", padx=10, pady=5
        )
        boton_reintentar.pack(pady=(10, 5))

        boton_salir = tk.Button(
            frame, text="Salir de la aplicación", command=cerrar,
            bg="#D32F2F", fg="white", font=("Segoe UI", 10, "bold"), relief="flat", padx=10, pady=5
        )
        boton_salir.pack()

        window.mainloop()

if __name__ == "__main__":
    mostrar_prediccion()
