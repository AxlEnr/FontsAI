import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from palette import selectColor
import sys
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Cargar datos
url = 'https://tinyurl.com/y2qmhfsr'
data = pd.read_csv(url)
X = data[['RED', 'GREEN', 'BLUE']].values / 255  
y = data['LIGHT_OR_DARK_FONT_IND'].values       

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear modelo MLP
mlp = MLPClassifier(hidden_layer_sizes=(8, 6), activation='relu', solver='sgd', max_iter=2000, random_state=42)

# Entrenar
mlp.fit(X_train, y_train)

# Evaluación
accuracy = mlp.score(X_test, y_test)
print(f"Precisión en test: {accuracy * 100:.2f}%")

# Predicción usando el modelo entrenado
def prediccion_fuente(color_rgb):
    color_np = np.array(color_rgb).reshape(1, -1) / 255
    return mlp.predict(color_np)

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
