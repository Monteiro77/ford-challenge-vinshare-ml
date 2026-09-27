"""
Inferência em lote: pontua uma base de veículos e gera a lista de leads por concessionária.

Uso:
    python src/predict.py data/veiculos_ford_vinshare.csv saida_leads.csv
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ml_utils import criar_features, limpar_dados  # noqa: E402

CAMINHO_MODELO = Path(__file__).resolve().parents[1] / "models" / "modelo_churn_vinshare.joblib"

ACOES = {
    "Alto": "Contato ativo do consultor em até 7 dias + oferta personalizada de revisão",
    "Médio": "Lembrete via app/WhatsApp com agendamento em 1 clique",
    "Baixo": "Manter régua de comunicação padrão",
}


def gerar_leads(df_bruto: pd.DataFrame, artefato: dict | None = None) -> pd.DataFrame:
    artefato = artefato or joblib.load(CAMINHO_MODELO)
    df = criar_features(limpar_dados(df_bruto))
    prob = artefato["pipeline"].predict_proba(df[artefato["features"]])[:, 1]

    faixa = np.select(
        [prob >= artefato["limiar"], prob >= artefato["limiar_medio"]], ["Alto", "Médio"], default="Baixo"
    )
    leads = pd.DataFrame({
        "vin": df["vin"].values,
        "concessionaria_id": df["concessionaria_id"].values,
        "modelo": df["modelo"].values,
        "meses_desde_ultimo_servico": df["meses_desde_ultimo_servico"].values,
        "garantia_ativa": df["garantia_ativa"].values,
        "prob_churn": prob.round(4),
        "faixa_risco": faixa,
    })
    leads["acao_sugerida"] = leads["faixa_risco"].map(ACOES)
    return leads.sort_values("prob_churn", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("Uso: python src/predict.py <entrada.csv> <saida.csv>")
    resultado = gerar_leads(pd.read_csv(sys.argv[1]))
    resultado.to_csv(sys.argv[2], index=False)
    print(resultado["faixa_risco"].value_counts().to_string())
    print(f"Leads salvos em {sys.argv[2]}")
