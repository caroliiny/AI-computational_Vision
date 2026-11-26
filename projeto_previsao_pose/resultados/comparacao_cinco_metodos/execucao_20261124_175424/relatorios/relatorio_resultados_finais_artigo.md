# Relatório final dos experimentos para o artigo

- Execução: `execucao_20261124_175424`
- Código identificado por hash de fontes: `ff82aec665915976e8a68c7811f19b63c800ac08e185c7273e47a60b80b85396`
- Árvore Git suja: `true`; portanto, registre este hash e preserve os fontes desta execução.
- Vídeo real: não foram calculadas métricas de desempenho real; a avaliação disponível utiliza pseudo-rótulos YOLO.

## Protocolos e dados

As poses são sintéticas, COCO-17, 2D. O split é feito por sequência antes da geração de janelas. Reconstrução offline usa contexto bidirecional e é avaliada somente nas articulações ocultas.
Previsão causal recebe dez quadros anteriores com ruído observacional, incluindo as máscaras de oclusão configuradas. A oclusão começa nos últimos cinco quadros do histórico; pode persistir no futuro e terminar com a reaparência das articulações. Os quadros posteriores ao corte são usados exclusivamente como referência para avaliação.

Condições de oclusão de teste: articulacoes_distais, membro_completo.
A condição `articulacoes_distais` oculta simultaneamente as articulações configuradas; `membro_completo` oculta um braço ou perna inteiro. As condições são agregadas separadamente.

## Configuração

Consulte [`configuracao_execucao.yaml`](configuracao_execucao.yaml), [`metadados.json`](../metadados.json) e [`tabela_1_configuracao.md`](../tabelas/tabela_1_configuracao.md). A Tabela 1 inclui versões e parâmetros efetivamente registrados.

## Resultados

- Reconstrução, articulacoes_distais_5_quadros: menor erro euclidiano médio observado para EKF (2.325 ± 0.000 px).
- Reconstrução, membro_completo_5_quadros: menor erro euclidiano médio observado para EKF (2.590 ± 0.000 px).
- Previsão futura, 10_quadros_futuros_oclusao_articulacoes_distais_5_quadros: menor erro euclidiano médio observado para EKF (6.173 ± 0.000 px).
- Previsão futura, 10_quadros_futuros_oclusao_membro_completo_5_quadros: menor erro euclidiano médio observado para EKF (6.567 ± 0.000 px).
- Previsão futura, 20_quadros_futuros_oclusao_articulacoes_distais_5_quadros: menor erro euclidiano médio observado para EKF (10.987 ± 0.000 px).
- Previsão futura, 20_quadros_futuros_oclusao_membro_completo_5_quadros: menor erro euclidiano médio observado para EKF (11.914 ± 0.000 px).
- Previsão futura, 5_quadros_futuros_oclusao_articulacoes_distais_5_quadros: menor erro euclidiano médio observado para EKF (4.573 ± 0.000 px).
- Previsão futura, 5_quadros_futuros_oclusao_membro_completo_5_quadros: menor erro euclidiano médio observado para EKF (4.712 ± 0.000 px).

### Menor erro por cenário e métrica

- articulacoes_distais_5_quadros: menor MAE para EKF (1.474 ± 0.000 px).
- articulacoes_distais_5_quadros: menor RMSE para EKF (1.865 ± 0.000 px).
- articulacoes_distais_5_quadros: menor erro euclidiano para EKF (2.325 ± 0.000 px).
- membro_completo_5_quadros: menor MAE para EKF (1.680 ± 0.000 px).
- membro_completo_5_quadros: menor RMSE para EKF (2.160 ± 0.000 px).
- membro_completo_5_quadros: menor erro euclidiano para EKF (2.590 ± 0.000 px).
- Em articulacoes_distais, a diferença entre as médias de erro euclidiano em 5 e 5 quadros para EKF foi +0.000 px. As máscaras são geradas separadamente por duração; não interprete essa diferença como efeito causal isolado da duração.
- Em articulacoes_distais, a diferença entre as médias de erro euclidiano em 5 e 5 quadros para LSTM foi +0.000 px. As máscaras são geradas separadamente por duração; não interprete essa diferença como efeito causal isolado da duração.
- Em articulacoes_distais, a diferença entre as médias de erro euclidiano em 5 e 5 quadros para GRU foi +0.000 px. As máscaras são geradas separadamente por duração; não interprete essa diferença como efeito causal isolado da duração.
- Em articulacoes_distais, a diferença entre as médias de erro euclidiano em 5 e 5 quadros para Transformer foi +0.000 px. As máscaras são geradas separadamente por duração; não interprete essa diferença como efeito causal isolado da duração.
- Em articulacoes_distais, a diferença entre as médias de erro euclidiano em 5 e 5 quadros para T-GNN foi +0.000 px. As máscaras são geradas separadamente por duração; não interprete essa diferença como efeito causal isolado da duração.
- Em membro_completo, a diferença entre as médias de erro euclidiano em 5 e 5 quadros para EKF foi +0.000 px. As máscaras são geradas separadamente por duração; não interprete essa diferença como efeito causal isolado da duração.
- Em membro_completo, a diferença entre as médias de erro euclidiano em 5 e 5 quadros para LSTM foi +0.000 px. As máscaras são geradas separadamente por duração; não interprete essa diferença como efeito causal isolado da duração.
- Em membro_completo, a diferença entre as médias de erro euclidiano em 5 e 5 quadros para GRU foi +0.000 px. As máscaras são geradas separadamente por duração; não interprete essa diferença como efeito causal isolado da duração.
- Em membro_completo, a diferença entre as médias de erro euclidiano em 5 e 5 quadros para Transformer foi +0.000 px. As máscaras são geradas separadamente por duração; não interprete essa diferença como efeito causal isolado da duração.
- Em membro_completo, a diferença entre as médias de erro euclidiano em 5 e 5 quadros para T-GNN foi +0.000 px. As máscaras são geradas separadamente por duração; não interprete essa diferença como efeito causal isolado da duração.
- Na previsão, compare os horizontes como médias ao longo de todos os quadros previstos; o valor do horizonte longo não é apenas o erro no último quadro.
- A variabilidade entre seeds é reportada nas tabelas, mas as seeds compartilham as mesmas sequências. Ela não mede generalização para novas identidades.
- As diferenças observadas entre T-GNN e os demais métodos não isolam causalmente o efeito da topologia anatômica.

### Previsão por horizonte

- 10_quadros_futuros_oclusao_articulacoes_distais_5_quadros: menor erro euclidiano médio para EKF (6.173 ± 0.000 px).
- 10_quadros_futuros_oclusao_membro_completo_5_quadros: menor erro euclidiano médio para EKF (6.567 ± 0.000 px).
- 20_quadros_futuros_oclusao_articulacoes_distais_5_quadros: menor erro euclidiano médio para EKF (10.987 ± 0.000 px).
- 20_quadros_futuros_oclusao_membro_completo_5_quadros: menor erro euclidiano médio para EKF (11.914 ± 0.000 px).
- 5_quadros_futuros_oclusao_articulacoes_distais_5_quadros: menor erro euclidiano médio para EKF (4.573 ± 0.000 px).
- 5_quadros_futuros_oclusao_membro_completo_5_quadros: menor erro euclidiano médio para EKF (4.712 ± 0.000 px).

A média e o desvio padrão são calculados entre seeds, após a média por sequência. Seeds compartilham as partições e sequências; o desvio padrão não estima generalização para novas pessoas ou vídeos.

Tabelas formatadas: [reconstrução](../tabelas/tabela_2_reconstrucao.md), [previsão](../tabelas/tabela_3_previsao.md) e [menor média observada por cenário/métrica](../tabelas/tabela_4_melhor_por_cenario.md). Resultados detalhados por sequência estão nos CSVs em `metricas/`: 30 linhas de reconstrução e 90 linhas de previsão.

## Figuras

- `grafico_1_reconstrucao.png`: erro versus duração, separado por condição.
- `grafico_2_erro_por_quadro.png`: erro ao longo da lacuna; área sombreada marca o intervalo ocluído.
- `grafico_3_previsao.png`: erro agregado por horizonte.
- `grafico_4_erro_por_horizonte.png`: erro em cada quadro da previsão de horizonte máximo.
- `figuras/`: pose de referência, entrada com oclusão e estimativas dos cinco métodos.

## Interpretação e limitações

Os achados limitam-se às sequências sintéticas, às condições configuradas e às arquiteturas/hiperparâmetros desta execução. Não permitem concluir superioridade universal, robustez a oclusões naturais ou desempenho em vídeo real. A previsão sintética avalia as condições configuradas; seus resultados não devem ser generalizados para oclusões naturais.

## Conteúdo a inserir no artigo

- **Metodologia:** dados sintéticos, split por sequência, tabela 1, definição dos protocolos e métricas.
- **Resultados e discussão:** tabelas 2 e 3, gráficos 1 a 4 e interpretação restrita aos cenários observados.
- **Conclusão:** síntese dos resultados sem vencedor universal, limitação da base procedural e necessidade de avaliação com anotações independentes em vídeos reais.

## Vídeo real

A etapa exploratória histórica utilizou um clipe de 754 quadros, 50 FPS, 768 × 432, processado pelo YOLOv8-Pose `yolov8n-pose.pt` (Ultralytics 8.4.204; confiança 0,45). Keypoints com baixa confiança foram interpolados dentro dos splits; cerca de 65,2% foram aceitos diretamente. A partição foi temporal no mesmo clipe/identidade, com embargo de 10 quadros. Não há anotação humana independente: portanto, nenhuma métrica desse clipe é apresentada aqui como acurácia real nem combinada com os resultados sintéticos. O valor histórico atípico do EKF em oclusão curta deve ser tratado como diagnóstico inconclusivo.

## Arquivos

Os dados, previsões, modelos, históricos, métricas, gráficos, figuras, vídeos, configuração e metadados ficam nas pastas desta execução. O detalhamento por sequência e seed é preservado em CSV.
