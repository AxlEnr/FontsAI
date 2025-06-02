import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import tkinter as tk
from palette import selectColor
import sys

def mostrar_prediccion(modelo):
    while True:
        color_rgb = selectColor()
        if not color_rgb:
            break

        color_list = list(map(int, color_rgb))
        color_predicho = modelo.predict([color_list])
        color_hex = '#%02x%02x%02x' % tuple(color_list)

        window = tk.Tk()
        window.title("Recomendación de Fuente")
        window.geometry("400x260")
        window.configure(bg="#f4f4f4")
        window.iconbitmap('assets/icon/icono.ico')
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

url = 'https://tinyurl.com/y2qmhfsr'
all_data = pd.read_csv(url)
X = all_data[['RED', 'GREEN', 'BLUE']]
y = all_data['LIGHT_OR_DARK_FONT_IND']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo = LogisticRegression()
modelo.fit(X_train, y_train)

mostrar_prediccion(modelo)
