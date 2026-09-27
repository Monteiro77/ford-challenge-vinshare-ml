"""Funções compartilhadas entre o notebook de treino e o script de inferência."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

MODELOS_CANONICOS = {
    m.lower(): m
    for m in ["Ranger", "Territory", "Maverick", "Bronco Sport", "Transit",
              "Mustang", "Ranger Raptor", "EcoSport", "Ka"]
}
TIPOS_CLIENTE = {"pf": "PF", "pj": "PJ", "frota": "Frota"}

FEATURES_NUMERICAS = [
    "idade_veiculo", "km_rodados", "km_por_ano", "meses_desde_ultimo_servico",
    "servicos_rede_24m", "ticket_medio", "distancia_concessionaria_km",
    "nps_ultima_visita", "alertas_telemetria_90d", "pct_lembretes_abertos",
    "reclamacoes_12m", "vin_share_hist_concessionaria",
]
FEATURES_BINARIAS = ["garantia_ativa", "app_conectado", "recall_pendente", "revisao_atrasada"]
FEATURES_CATEGORICAS = ["modelo", "tipo_cliente", "pais"]
FEATURES = FEATURES_NUMERICAS + FEATURES_BINARIAS + FEATURES_CATEGORICAS
ALVO = "churn_12m"


def limpar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """Corrige inconsistências por regras de domínio (não depende de estatísticas dos dados)."""
    df = df.copy()
    df = df.drop_duplicates(subset="vin", keep="first")

    df["modelo"] = df["modelo"].astype(str).str.strip().str.lower().map(MODELOS_CANONICOS)
    df["tipo_cliente"] = df["tipo_cliente"].str.strip().str.lower().map(TIPOS_CLIENTE)

    df["nps_ultima_visita"] = pd.to_numeric(df["nps_ultima_visita"], errors="coerce")
    df.loc[~df["nps_ultima_visita"].between(0, 10), "nps_ultima_visita"] = np.nan

    # Valores fisicamente impossíveis viram ausentes e são imputados depois
    df.loc[(df["km_rodados"] < 0) | (df["km_rodados"] >= 900000), "km_rodados"] = np.nan
    df.loc[df["idade_veiculo"] < 0, "idade_veiculo"] = np.nan
    return df


def criar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["km_por_ano"] = df["km_rodados"] / (df["idade_veiculo"] + 0.5)
    # Plano de manutenção Ford: revisão a cada 12 meses ou 10.000 km
    km_desde_servico = df["km_por_ano"] * df["meses_desde_ultimo_servico"] / 12
    df["revisao_atrasada"] = ((df["meses_desde_ultimo_servico"] > 12) | (km_desde_servico > 10000)).astype(int)
    return df


class Winsorizer(BaseEstimator, TransformerMixin):
    """Limita cada coluna aos quantis aprendidos no treino (tratamento de outliers sem vazamento)."""

    def __init__(self, inferior=0.01, superior=0.99):
        self.inferior = inferior
        self.superior = superior

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lim_inf_ = np.nanquantile(X, self.inferior, axis=0)
        self.lim_sup_ = np.nanquantile(X, self.superior, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lim_inf_, self.lim_sup_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)
