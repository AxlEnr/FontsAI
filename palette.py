import tkinter as tk
from tkinter import colorchooser

def selectColor():
    color_resultado = None

    def abrir_selector():
        nonlocal color_resultado
        color = colorchooser.askcolor(title="Selecciona un color")
        if color[1]:
            color_resultado = color[0]
            ventana.quit()

    ventana = tk.Tk()
    ventana.title("🎨 Paleta de Colores")
    ventana.geometry("300x120")
    ventana.configure(bg="#f0f0f0")
    ventana.iconbitmap('assets/icon/icono.ico')
    ventana.overrideredirect(True)


    frame = tk.Frame(ventana, bg="#ffffff", bd=2, relief="groove")
    frame.place(relx=0.5, rely=0.5, anchor="center", width=300, height=120)

    titulo = tk.Label(frame, text="Selecciona un color", font=("Segoe UI", 14, "bold"), bg="#ffffff")
    titulo.pack(pady=10)

    boton = tk.Button(frame, text="Abrir selector", command=abrir_selector,
                      bg="#4CAF50", fg="white", font=("Segoe UI", 10, "bold"), relief="flat", padx=10, pady=5)
    boton.pack()

    ventana.mainloop()
    ventana.destroy()
    return color_resultado
