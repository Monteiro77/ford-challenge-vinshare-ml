"""
Gerador de dados sintéticos para o Desafio 02 Ford - VIN Share (América do Sul).

Cada linha representa um veículo Ford (identificado pelo VIN) na data de corte da
análise. O alvo `churn_12m` indica se o veículo NÃO realizou nenhum serviço na rede
oficial Ford nos 12 meses seguintes à data de corte (ou seja, saiu da rede).

A geração segue regras de negócio plausíveis (garantia, tempo sem revisão, distância,
preço, engajamento com o app etc.) somadas a ruído aleatório, e depois insere
"sujeira" proposital (valores ausentes, inconsistências, outliers e duplicatas) para
exercitar a etapa de preparação dos dados.

Uso:  python src/gerar_dados.py  -> gera data/veiculos_ford_vinshare.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ANO_REFERENCIA = 2026

MODELOS = {
    # modelo: (probabilidade, idade média, é legado?, taxa de app conectado)
    "Ranger": (0.26, 3.5, False, 0.55),
    "Territory": (0.14, 2.0, False, 0.75),
    "Maverick": (0.12, 2.0, False, 0.70),
    "Bronco Sport": (0.08, 2.5, False, 0.70),
    "Transit": (0.07, 3.0, False, 0.45),
    "Mustang": (0.03, 3.0, False, 0.65),
    "Ranger Raptor": (0.03, 1.5, False, 0.80),
    "EcoSport": (0.15, 7.0, True, 0.20),
    "Ka": (0.12, 8.0, True, 0.10),
}

PAISES = {
    # país: (probabilidade, prefixo VIN, nº de concessionárias)
    "Brasil": (0.55, "9BF", 45),
    "Argentina": (0.20, "8AF", 15),
    "Chile": (0.10, "MNB", 8),
    "Colômbia": (0.08, "3FA", 6),
    "Peru": (0.07, "MPB", 6),
}

VIN_CHARS = np.array(list("ABCDEFGHJKLMNPRSTUVWXYZ0123456789"))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def gerar(n: int = 20000, seed: int = 42, taxa_churn_alvo: float = 0.30) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ---------------- Identificação e contexto ----------------
    paises = list(PAISES)
    pais = rng.choice(paises, n, p=[PAISES[p][0] for p in paises])

    vin = np.array(
        [PAISES[p][1] + "".join(rng.choice(VIN_CHARS, 14)) for p in pais]
    )

    # Concessionárias e o VIN Share histórico de cada uma
    concessionaria = np.empty(n, dtype=object)
    share_hist = np.empty(n)
    for p in paises:
        idx = np.where(pais == p)[0]
        n_dealers = PAISES[p][2]
        dealers = [f"{PAISES[p][1]}-D{d:03d}" for d in range(1, n_dealers + 1)]
        dealer_share = dict(zip(dealers, rng.beta(6, 4, n_dealers)))
        escolhidos = rng.choice(dealers, len(idx))
        concessionaria[idx] = escolhidos
        share_hist[idx] = [dealer_share[d] for d in escolhidos]

    modelos = list(MODELOS)
    modelo = rng.choice(modelos, n, p=[MODELOS[m][0] for m in modelos])
    idade_media = np.array([MODELOS[m][1] for m in modelo])
    legado = np.array([MODELOS[m][2] for m in modelo])
    taxa_app = np.array([MODELOS[m][3] for m in modelo])

    tipo_cliente = rng.choice(["PF", "PJ", "Frota"], n, p=[0.70, 0.18, 0.12])

    # ---------------- Veículo ----------------
    idade = np.clip(np.round(rng.gamma(4, idade_media / 4)), 0, 15).astype(int)
    km_ano_base = np.where(tipo_cliente == "Frota", 38000,
                           np.where(tipo_cliente == "PJ", 25000, 14000))
    km_rodados = np.clip(
        (idade + rng.uniform(0.2, 1.0, n)) * rng.normal(km_ano_base, km_ano_base * 0.3),
        500, None,
    ).round(0)

    # Garantia de fábrica de 3 anos + 15% com garantia estendida até 5 anos
    estendida = rng.random(n) < 0.15
    garantia_ativa = ((idade <= 3) | (estendida & (idade <= 5))).astype(int)

    recall_pendente = (rng.random(n) < 0.08).astype(int)

    # ---------------- Relacionamento com a rede ----------------
    servicos_rede_24m = rng.poisson(np.clip(3.2 - 0.25 * idade + 1.0 * garantia_ativa, 0.2, None))
    meses_desde_ultimo_servico = np.clip(
        np.round(rng.gamma(2.0, 3.0 + 0.8 * idade - 1.5 * garantia_ativa.clip(0, 1) + 0.5)), 0, 72
    ).astype(int)
    ticket_medio = np.clip(rng.normal(750 + 65 * idade, 220), 150, None).round(2)
    distancia = np.clip(rng.lognormal(np.log(12), 0.8, n), 0.5, 250).round(1)
    nps = np.clip(np.round(rng.normal(7.4 + 0.8 * (share_hist - 0.6) * 5, 1.9)), 0, 10)
    app_conectado = (rng.random(n) < taxa_app).astype(int)
    alertas = np.where(app_conectado == 1, rng.poisson(0.6, n), 0)
    pct_lembretes = np.clip(rng.beta(2, 3, n) + 0.15 * app_conectado, 0, 1).round(3)
    reclamacoes = rng.poisson(0.25, n)

    # ---------------- Variável alvo (regra latente + ruído) ----------------
    z = (
        1.10 * (1 - garantia_ativa)
        + 0.06 * (meses_desde_ultimo_servico - 8)
        + 0.90 * (meses_desde_ultimo_servico > 14)           # efeito limiar (revisão muito atrasada)
        - 0.32 * np.minimum(servicos_rede_24m, 5)
        + 0.55 * np.log(distancia / 12)
        + 0.0020 * (ticket_medio - 900) * (1 - garantia_ativa)  # sensibilidade a preço fora da garantia
        - 0.20 * (nps - 7)
        - 0.55 * app_conectado
        - 0.30 * alertas                                      # alertas de telemetria trazem o cliente
        - 1.20 * (pct_lembretes - 0.4)
        + 0.45 * reclamacoes
        - 0.60 * recall_pendente
        + np.where(tipo_cliente == "Frota", 0.35, np.where(tipo_cliente == "PJ", -0.15, 0))
        + 0.45 * ((tipo_cliente == "Frota") & (km_rodados > 150000))  # frotas rodadas vão para oficinas próprias
        + 0.40 * legado
        - 0.30 * np.isin(modelo, ["Mustang", "Ranger Raptor"])
        - 2.00 * (share_hist - 0.6)
        + 0.05 * (idade - 4)
        + rng.normal(0, 1.0, n)
    )
    # Ajusta o intercepto para atingir a taxa de churn desejada
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _sigmoid(z + mid).mean() > taxa_churn_alvo:
            hi = mid
        else:
            lo = mid
    prob = _sigmoid(z + (lo + hi) / 2)
    churn = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame({
        "vin": vin,
        "pais": pais,
        "concessionaria_id": concessionaria,
        "modelo": modelo,
        "ano_modelo": ANO_REFERENCIA - idade,
        "idade_veiculo": idade.astype(float),
        "km_rodados": km_rodados,
        "tipo_cliente": tipo_cliente,
        "garantia_ativa": garantia_ativa,
        "recall_pendente": recall_pendente,
        "servicos_rede_24m": servicos_rede_24m,
        "meses_desde_ultimo_servico": meses_desde_ultimo_servico,
        "ticket_medio": ticket_medio,
        "distancia_concessionaria_km": distancia,
        "nps_ultima_visita": nps.astype(object),
        "app_conectado": app_conectado,
        "alertas_telemetria_90d": alertas,
        "pct_lembretes_abertos": pct_lembretes,
        "reclamacoes_12m": reclamacoes,
        "vin_share_hist_concessionaria": share_hist.round(3),
        "churn_12m": churn,
    })

    # ---------------- "Sujeira" proposital ----------------
    def amostra(frac):
        return rng.choice(n, int(n * frac), replace=False)

    # Inconsistência de grafia nas categorias
    i = amostra(0.06)
    variacoes = [str.upper, str.lower, lambda s: f"  {s} "]
    df.loc[i, "modelo"] = [variacoes[k % 3](m) for k, m in enumerate(df.loc[i, "modelo"])]
    i = amostra(0.02)
    df.loc[i, "tipo_cliente"] = df.loc[i, "tipo_cliente"].str.lower()

    # Valores impossíveis e outliers
    df.loc[amostra(0.005), "km_rodados"] = -df["km_rodados"].sample(int(n * 0.005), random_state=1).values
    df.loc[amostra(0.003), "km_rodados"] = 999999
    df.loc[amostra(0.002), "idade_veiculo"] = -1
    i = amostra(0.005)
    df.loc[i, "ticket_medio"] = (df.loc[i, "ticket_medio"] * rng.uniform(5, 12, len(i))).round(2)
    df.loc[amostra(0.003), "nps_ultima_visita"] = "N/A"
    df.loc[amostra(0.002), "nps_ultima_visita"] = 11

    # Valores ausentes
    for col, frac in {
        "nps_ultima_visita": 0.12,   # cliente não respondeu pesquisa
        "ticket_medio": 0.03,
        "distancia_concessionaria_km": 0.04,
        "pct_lembretes_abertos": 0.06,
        "km_rodados": 0.02,
        "tipo_cliente": 0.01,
        "alertas_telemetria_90d": 0.03,
    }.items():
        df.loc[amostra(frac), col] = np.nan

    # Registros duplicados (falha de integração entre sistemas)
    dup = df.sample(frac=0.01, random_state=seed)
    df = pd.concat([df, dup]).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    destino = Path(__file__).resolve().parents[1] / "data" / "veiculos_ford_vinshare.csv"
    destino.parent.mkdir(exist_ok=True)
    dados = gerar()
    dados.to_csv(destino, index=False)
    print(f"{len(dados)} registros salvos em {destino}")
    print(f"Taxa de churn: {dados['churn_12m'].mean():.1%}")
