| Parâmetro | Configuração |
| --- | --- |
| Pose | COCO-17, 17 articulações, 2D |
| Sequências/quadro | 12 / 40 |
| Split | 70%, 15%, 15%; contagens={"treino": 8, "validacao": 1, "teste": 3} |
| Seeds | 11 |
| Oclusões de teste | 5 |
| Ruído observacional | 1.5 px (desvio padrão) |
| Previsão | 10 quadros históricos; horizontes [5, 10, 20]; histórico com oclusões, seguido por quadros futuros usados somente como referência |
| Arquiteturas | LSTM/GRU: recorrência bidirecional, 2 camadas; Transformer: encoder temporal, 2 camadas/4 cabeças; T-GNN: 2 camadas espaciais COCO-17 e GRU bidirecional. As redes usam dimensão oculta configurada. |
| Redes | Adam, lr=0.001, batch=16, máximo 2 épocas, early stopping=2; dimensão oculta=64; loss MSE (reconstrução: peso ocluso 2.0, previsão: MSE) |
| Seleção | Reconstrução: menor MAE de validação nos pontos ocultos; previsão: menor média da MAE de validação entre horizontes |
| EKF | Ruídos de processo ajustados por bootstrap no treino; valores por seed em metadados.json: {"11": {"q_v": 1.7579373121261597, "q_w": 1.0, "tau": 1.5, "q_pos": 0.05, "q_th": 0.005}} |
| Dispositivo | cpu |
| Python/PyTorch | 3.12.10 / 2.15.0+cpu |
| Pacotes | {"numpy": "2.5.4", "matplotlib": "3.11.2", "opencv": "5.0.0", "pyyaml": "6.0.3"} |