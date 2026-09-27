# Ford Challenge 2026: Sprint 3 de IA & Machine Learning
### Desafio 02: Impulsionando o VIN Share na América do Sul

**Integrantes**

| Integrante | RM |
|---|---|
| Vinicius Monteiro Araújo | 555088 |
| Guilherme Almeida | 555180 |
| Rafael Duarte de Freitas | 558644 |
| Rafael Gaspar Bragança Martins | 557228 |
| Luiz Gustavo da Silva | 558358 |

## A solução em uma frase
Um modelo de classificação estima, para cada veículo Ford, a **probabilidade de ele não voltar à rede oficial nos próximos 12 meses**. A base vira uma **lista de leads priorizada por concessionária**, com uma ação sugerida para cada faixa de risco.

## Problema
- **Indicador de negócio:** VIN Share, o % de veículos Ford que fazem manutenção na rede oficial.
- **Pergunta de ML:** *este VIN vai deixar de frequentar a rede nos próximos 12 meses?*
- **Tipo:** aprendizado supervisionado, **classificação binária** com classes desbalanceadas (~30% de churn).
- **Dados:** 20.200 registros **sintéticos** (`src/gerar_dados.py`) com 20 variáveis de cadastro, garantia, histórico de serviços, telemetria/app, satisfação e concessionária. A sujeira foi inserida de propósito: duplicatas, grafias inconsistentes, valores impossíveis, outliers e ~12% de NPS ausente.

## Preparação dos dados
| Etapa | O que foi feito |
|---|---|
| Regras de domínio | remoção de 200 VINs duplicados, padronização de `modelo`/`tipo_cliente`, valores impossíveis (km < 0, km = 999.999, idade = −1, NPS fora de 0–10 ou "N/A") → ausentes |
| Features novas | `km_por_ano` e `revisao_atrasada` (> 12 meses ou > 10.000 km desde a última visita) |
| Outliers | winsorização nos percentis 1%/99% aprendidos no treino |
| Ausentes | mediana/moda + indicador de ausência |
| Transformações | `StandardScaler` (numéricas), `OneHotEncoder` (categóricas) |
| Divisão | 80/20 estratificada. Tudo que é estatístico fica dentro de um `Pipeline`, ajustado só no treino (sem vazamento) |

## Modelos comparados
Seis algoritmos, com ajuste de hiperparâmetros (Grid/RandomizedSearchCV, otimizando PR-AUC):

| Modelo | PR-AUC CV 5-fold | ROC-AUC teste | PR-AUC teste | Captura top-20% |
|---|---|---|---|---|
| **Regressão Logística (L1, C=0,1)** ✅ | **0,816 ± 0,010** | **0,910** | 0,836 | 57,8% |
| Gradient Boosting (HistGB) | 0,815 ± 0,012 | 0,909 | 0,837 | 58,4% |
| Rede Neural MLP (32, tanh, α=0,1) | 0,812 ± 0,015 | 0,907 | 0,833 | 57,3% |
| Random Forest | 0,805 ± 0,011 | 0,903 | 0,828 | 57,8% |
| KNN (k=51, distance) | 0,793 ± 0,011 | 0,899 | 0,819 | 57,8% |
| Árvore de Decisão (depth=8) | 0,769 ± 0,016 | 0,889 | 0,792 | 55,1% |

**Métrica principal:** PR-AUC, a mais adequada para a classe minoritária e para ranquear leads. A acurácia é só referência, porque prever "ninguém sai" já daria 70%.

## Modelo final e justificativa
**Regressão Logística.** Teve o maior PR-AUC médio na validação cruzada, empatada tecnicamente com o Gradient Boosting e a MLP. Entre modelos equivalentes, escolhemos o **mais estável, leve e explicável**: cada coeficiente é um argumento de negócio para o consultor, e isso também atende à transparência exigida pela LGPD para decisões automatizadas.

Com o limiar ajustado (0,61, escolhido por F1 *out-of-fold*), no teste:
- **Recall de 78%** e **precision de 75%** na classe churn.
- A faixa **Alto risco** reúne 31% da base e **78% dos churns** (taxa real de 75%, contra 5% na faixa Baixo).

**Fatores que mais aumentam o risco:** meses desde o último serviço, cliente de frota, modelos legados (EcoSport/Ka), distância até a concessionária, ticket médio alto e reclamações.
**Fatores que mais protegem:** garantia ativa, serviços recentes na rede, app FordPass conectado, recall pendente, NPS alto e lembretes abertos.

## Deploy proposto
1. **Lote:** um job semanal pontua toda a base (`src/predict.py`) e publica os leads para o app do consultor e para o dashboard de VIN Share.
2. **Online:** um microsserviço FastAPI com JWT (`POST /v1/veiculos/{vin}/risco-churn`), acionado pelo app ou por alertas de telemetria.
3. **MLOps:** versionamento (MLflow), monitoramento de drift (PSI), acompanhamento da conversão dos leads e retreino trimestral.

## Trabalhos futuros
Retreinar com dados reais e validação temporal, *uplift modeling* (quem responde à ação), análise de sobrevivência (quando o cliente sai), explicações SHAP por lead, limiar orientado a custo e teste A/B das ações de retenção.

## Estrutura
```
├── data/veiculos_ford_vinshare.csv          # dados sintéticos
├── notebooks/sprint3_ia_ml_vinshare.ipynb   # notebook completo, já executado (entregável principal)
├── notebooks/sprint3_ia_ml_vinshare.html    # mesmo notebook em HTML
├── models/modelo_churn_vinshare.joblib      # pipeline treinado + limiar
├── models/comparacao_modelos.csv            # métricas de teste de todos os modelos
└── src/
    ├── gerar_dados.py   # gerador dos dados sintéticos
    ├── ml_utils.py      # limpeza, features e Winsorizer (compartilhado treino/inferência)
    └── predict.py       # inferência em lote -> lista de leads
```

## Como executar
```bash
pip install -r requirements.txt
python src/gerar_dados.py                  # (opcional) regenera os dados, seed 42
jupyter notebook notebooks/sprint3_ia_ml_vinshare.ipynb
python src/predict.py data/veiculos_ford_vinshare.csv leads.csv   # gera a lista de leads
```
