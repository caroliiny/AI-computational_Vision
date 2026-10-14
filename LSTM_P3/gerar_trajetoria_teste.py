
import csv
import math
import random

random.seed(42)

quantidade_quadros = 200
trajetoria = []

x = 100.0
y = 100.0
vx = 3.0
vy = 1.5

for frame in range(quantidade_quadros):
    # Pequenas mudanças de direção e velocidade
    vx += random.uniform(-0.8, 0.8)
    vy += random.uniform(-0.8, 0.8)

    # Limita a velocidade para manter o alvo na cena
    vx = max(-5, min(5, vx))
    vy = max(-5, min(5, vy))

    x += vx
    y += vy

    # Mantém o alvo dentro de uma área de 800 x 600
    x = max(20, min(780, x))
    y = max(20, min(580, y))

    # Oclusão artificial entre os quadros 80 e 109
    visivel = 0 if 80 <= frame <= 109 else 1

    trajetoria.append({
        "frame": frame,
        "x_real": round(x, 2),
        "y_real": round(y, 2),
        "visivel": visivel
    })

with open(
    "trajetoria_teste.csv",
    "w",
    newline="",
    encoding="utf-8"
) as arquivo:
    campos = ["frame", "x_real", "y_real", "visivel"]
    escritor = csv.DictWriter(arquivo, fieldnames=campos)

    escritor.writeheader()
    escritor.writerows(trajetoria)

print("Trajetória de teste criada!")
print("Arquivo: trajetoria_teste.csv")
print("Quadros gerados:", quantidade_quadros)
print("Oclusão simulada: quadros 80 a 109")
