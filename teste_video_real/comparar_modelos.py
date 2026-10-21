import numpy as np
import matplotlib.pyplot as plt

# 1. Carregar os dados extraídos do vídeo
print("A carregar 'pose_data.npz'...")
data = np.load('pose_data.npz')
kpts = data['keypoints']      # Shape: (N_frames, 17, 2)
vis = data['visibility']       # Shape: (N_frames, 17)

def simular_ekf_predicao(kpts_seq, vis_seq, max_vel=15.0):
    """Simula o EKF mantendo a última posição/velocidade válida conhecida."""
    N, J, _ = kpts_seq.shape
    kpts_pred = np.copy(kpts_seq)

    for j in range(J):
        last_valid_pos = None
        last_valid_vel = np.array([0.0, 0.0])

        for t in range(N):
            # Verifica se a junta é visível e não é uma coordenada nula (0,0)
            if vis_seq[t, j] == 1 and not (kpts_seq[t, j, 0] == 0 and kpts_seq[t, j, 1] == 0):
                if last_valid_pos is not None:
                    # Calcula a velocidade real limitada
                    vel = kpts_seq[t, j] - last_valid_pos
                    last_valid_vel = np.clip(vel, -max_vel, max_vel)
                last_valid_pos = kpts_seq[t, j].astype(float).copy()
                kpts_pred[t, j] = kpts_seq[t, j]
            else:
                # Junta ocluída: projeta a partir da última posição e velocidade válidas
                if last_valid_pos is not None:
                    last_valid_pos += last_valid_vel
                    kpts_pred[t, j] = last_valid_pos.copy()
                    # Reduz a velocidade gradualmente (0.90) durante a oclusão
                    last_valid_vel *= 0.90
                else:
                    kpts_pred[t, j] = np.array([0.0, 0.0])

    return kpts_pred

print("A calcular predições cinemáticas do EKF...")
kpts_ekf = simular_ekf_predicao(kpts, vis)

# 2. Cálculo do Jittering (Estabilidade de movimento)
diff_real = np.diff(kpts, axis=0)
diff_ekf = np.diff(kpts_ekf, axis=0)

jitter_real = np.mean(np.var(diff_real, axis=0))
jitter_ekf = np.mean(np.var(diff_ekf, axis=0))
taxa_oclusao = (1 - np.mean(vis)) * 100

print("\n================ RESUMO MÉTRICO ================")
print(f"Total de quadros analisados: {len(kpts)}")
print(f"Taxa Média de Oclusão no Vídeo: {taxa_oclusao:.2f}%")
print(f"Jittering (Deteção Base YOLOv8): {jitter_real:.2f}")
print(f"Jittering (Filtro Cinemático EKF): {jitter_ekf:.2f}")
print("===================================================\n")

# 3. Gerar e guardar o gráfico da trajetória do pulso direito (Keypoint 10)
plt.figure(figsize=(12, 5))
plt.plot(kpts[:, 10, 0], label='Deteção Base (YOLOv8)', color='gray', linestyle='--')
plt.plot(kpts_ekf[:, 10, 0], label='Estimativa EKF', color='blue')
plt.title('Trajetória Horizontal do Pulso Direito Sob Oclusão dos Pilares')
plt.xlabel('Quadro (Frame)')
plt.ylabel('Posição X (Pixels)')
plt.legend()
plt.grid(True)

plt.savefig('trajetoria_pulso.png')
print(" Gráfico guardado com sucesso como 'trajetoria_pulso.png'!")
print("A abrir a janela do gráfico...")
plt.show()