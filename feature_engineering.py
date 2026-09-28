"""Audita leakage e gera features pré-largada a partir de dataset_limpo.csv.

Execute a partir de qualquer diretório com:
    python feature_engineering.py

O dataset limpo é tratado como fonte somente de leitura. A saída é uma matriz
separada, dataset_features.csv, com uma lista explícita de features permitidas.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "dataset_limpo.csv"
OUTPUT_PATH = ROOT / "dataset_features.csv"

CURRENT_SUMMARY_PATTERN = re.compile(
    r"^(driver|constructor|engmfr|tyremfr)_(total|best)_"
)

# Estes campos descrevem informação que pode ser conhecida antes da largada.
# Grid e classificação são dados de sessões anteriores à corrida; motor e pneus
# descrevem o carro inscrito no evento.
SAFE_PRESTART_RD_COLUMNS = {
    "rd_race_id",
    "rd_driver_id",
    "rd_constructor_id",
    "rd_driver_number",
    "rd_engine_manufacturer_id",
    "rd_tyre_manufacturer_id",
    "rd_race_grid_position_number",
    "rd_race_grid_position_text",
    "rd_race_qualification_position_number",
    "rd_race_qualification_position_text",
}

FEATURE_COLUMNS = [
    "race_year",
    "race_round",
    "rd_driver_id",
    "rd_constructor_id",
    "race_circuit_id",
    "rd_engine_manufacturer_id",
    "rd_tyre_manufacturer_id",
    "driver_age_years",
    "grid_position",
    "qualifying_position",
    "fp1_position",
    "fp2_position",
    "fp3_position",
    "race_circuit_layout_id",
    "race_course_length",
    "race_turns",
    "race_circuit_type",
    "race_direction",
    "driver_prior_entries",
    "driver_prior_wins",
    "driver_prior_win_rate",
    "driver_prior_podiums",
    "driver_prior_avg_finish_last5",
    "driver_prior_avg_points_last5",
    "driver_circuit_prior_entries",
    "driver_circuit_prior_wins",
    "driver_circuit_prior_win_rate",
    "constructor_prior_races",
    "constructor_prior_wins",
    "constructor_prior_win_rate",
    "constructor_prior_avg_finish",
    "constructor_prior_avg_points_last5",
]

TARGET_COLUMNS = [
    "target_finish_order",
    "target_classified_position",
    "target_winner",
]

METADATA_COLUMNS = [
    "rd_race_id",
    "race_date",
    "race_year",
    "race_round",
    "rd_driver_id",
    "rd_constructor_id",
    "race_circuit_id",
    "race_circuit_layout_id",
]

REQUIRED_COLUMNS = {
    "rd_race_id",
    "rd_driver_id",
    "rd_constructor_id",
    "rd_position_display_order",
    "rd_position_number",
    "rd_race_points",
    "rd_engine_manufacturer_id",
    "rd_tyre_manufacturer_id",
    "race_year",
    "race_round",
    "race_date",
    "race_circuit_id",
    "race_circuit_layout_id",
    "race_course_length",
    "race_turns",
    "race_circuit_type",
    "race_direction",
    "driver_date_of_birth",
}


def leakage_risk(column: str) -> str | None:
    """Return why a source column should not be used directly as a feature."""
    if column.startswith(("dstand_", "cstand_")):
        return "Standing atualizado após a corrida"
    if column in {
        "race_drivers_championship_decider",
        "race_constructors_championship_decider",
    }:
        return "Marcador de campeonato dependente do estado da temporada"
    if CURRENT_SUMMARY_PATTERN.match(column):
        return "Resumo atual de carreira/fabricante; pode incluir o futuro"
    if column in {"circuit_total_races_held", "gp_total_races_held"}:
        return "Contagem atual de eventos; pode incluir corridas futuras"
    if column.startswith(("engine_", "chassis_", "ed_")):
        return "Agregação por temporada sem vigência por etapa comprovada"
    if column.startswith("sess_fastest_lap_"):
        return "Volta mais rápida durante a corrida"
    if column.startswith("rd_") and column not in SAFE_PRESTART_RD_COLUMNS:
        return "Campo de resultado/sessão sem liberação explícita como pré-largada"
    if (
        column == "race_time"
        or column.startswith("race_scheduled_")
        or re.match(
            r"^race_(?:pre_qualifying|free_practice_[123]|qualifying)_(?:date|time)$",
            column,
        )
    ):
        return "Cobertura temporal parcial documentada na limpeza"
    return None


def audit_source_columns(columns: list[str]) -> pd.DataFrame:
    """Create a review table of known leakage and coverage risks."""
    rows = [
        {"coluna": column, "decisão": reason}
        for column in columns
        if (reason := leakage_risk(column)) is not None
    ]
    return pd.DataFrame(rows, columns=["coluna", "decisão"])


def _require_columns(frame: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Colunas esperadas ausentes em dataset_limpo.csv: {missing}")


def build_feature_frame(source: pd.DataFrame) -> pd.DataFrame:
    """Build a pre-race modelling table without changing or cleaning source rows."""
    _require_columns(source, REQUIRED_COLUMNS)

    if source.duplicated(["rd_race_id", "rd_driver_id"]).any():
        raise ValueError("A base limpa tem mais de uma linha por corrida e piloto.")

    work = source.copy()
    work["race_date"] = pd.to_datetime(work["race_date"], errors="coerce")
    if work["race_date"].isna().any():
        raise ValueError("Há datas de corrida inválidas; não é seguro ordenar o histórico.")

    work = work.sort_values(
        ["race_date", "race_year", "race_round", "rd_race_id", "rd_driver_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    work["_row_order"] = range(len(work))

    # Alvos, separados das features. A ordem completa também cobre DNF/DNS/DSQ.
    work["target_finish_order"] = pd.to_numeric(
        work["rd_position_display_order"], errors="coerce"
    )
    work["target_classified_position"] = pd.to_numeric(
        work["rd_position_number"], errors="coerce"
    )
    work["target_winner"] = work["target_finish_order"].eq(1).astype("int8")
    if work["target_finish_order"].isna().any():
        raise ValueError("A ordem oficial tem valores ausentes; não é possível criar o alvo completo.")

    # Valores de pontos ausentes indicam zero pontos. São usados só em históricos defasados.
    work["_points_for_history"] = pd.to_numeric(
        work["rd_race_points"], errors="coerce"
    ).fillna(0)

    # Histórico de piloto: toda estatística usa shift(1), nunca a corrida atual.
    driver_group = work.groupby("rd_driver_id", sort=False)
    work["driver_prior_entries"] = driver_group.cumcount()
    work["driver_prior_wins"] = driver_group["target_winner"].transform(
        lambda series: series.shift(1).fillna(0).cumsum()
    ).astype("int64")
    driver_entry_denominator = work["driver_prior_entries"].astype("float64").where(
        work["driver_prior_entries"].gt(0)
    )
    work["driver_prior_win_rate"] = (
        work["driver_prior_wins"] / driver_entry_denominator
    )
    work["driver_prior_podiums"] = driver_group["target_finish_order"].transform(
        lambda series: series.shift(1).le(3).astype("int64").cumsum()
    ).astype("int64")
    work["driver_prior_avg_finish_last5"] = driver_group[
        "target_finish_order"
    ].transform(
        lambda series: series.shift(1).rolling(window=5, min_periods=1).mean()
    )
    work["driver_prior_avg_points_last5"] = driver_group[
        "_points_for_history"
    ].transform(
        lambda series: series.shift(1).rolling(window=5, min_periods=1).mean()
    )

    # Histórico do piloto no circuito atual, também somente com GPs anteriores.
    circuit_group = work.groupby(
        ["rd_driver_id", "race_circuit_id"], sort=False
    )
    work["driver_circuit_prior_entries"] = circuit_group.cumcount()
    work["driver_circuit_prior_wins"] = circuit_group["target_winner"].transform(
        lambda series: series.shift(1).fillna(0).cumsum()
    ).astype("int64")
    circuit_entry_denominator = work["driver_circuit_prior_entries"].astype(
        "float64"
    ).where(work["driver_circuit_prior_entries"].gt(0))
    work["driver_circuit_prior_win_rate"] = (
        work["driver_circuit_prior_wins"] / circuit_entry_denominator
    )

    # Resume o construtor por corrida antes de calcular seu histórico. Isso impede
    # que o resultado de um companheiro na etapa atual vaze para o outro piloto.
    team_race = (
        work.groupby(
            ["rd_race_id", "rd_constructor_id"], as_index=False, sort=False
        )
        .agg(
            race_date=("race_date", "first"),
            race_year=("race_year", "first"),
            race_round=("race_round", "first"),
            team_wins=("target_winner", "max"),
            team_points=("_points_for_history", "sum"),
            team_avg_finish=("target_finish_order", "mean"),
        )
        .sort_values(
            ["race_date", "race_year", "race_round", "rd_race_id", "rd_constructor_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    team_group = team_race.groupby("rd_constructor_id", sort=False)
    team_race["constructor_prior_races"] = team_group.cumcount()
    team_race["constructor_prior_wins"] = team_group["team_wins"].transform(
        lambda series: series.shift(1).fillna(0).cumsum()
    ).astype("int64")
    team_race_denominator = team_race["constructor_prior_races"].astype(
        "float64"
    ).where(team_race["constructor_prior_races"].gt(0))
    team_race["constructor_prior_win_rate"] = (
        team_race["constructor_prior_wins"] / team_race_denominator
    )
    team_race["constructor_prior_avg_finish"] = team_group[
        "team_avg_finish"
    ].transform(
        lambda series: series.shift(1).expanding(min_periods=1).mean()
    )
    # A janela curta reduz a mistura entre sistemas de pontuação de épocas distintas.
    team_race["constructor_prior_avg_points_last5"] = team_group[
        "team_points"
    ].transform(
        lambda series: series.shift(1).rolling(window=5, min_periods=1).mean()
    )

    team_feature_columns = [
        "constructor_prior_races",
        "constructor_prior_wins",
        "constructor_prior_win_rate",
        "constructor_prior_avg_finish",
        "constructor_prior_avg_points_last5",
    ]
    work = work.merge(
        team_race[["rd_race_id", "rd_constructor_id"] + team_feature_columns],
        on=["rd_race_id", "rd_constructor_id"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    work = work.sort_values("_row_order", kind="mergesort").reset_index(drop=True)

    # Dados disponíveis imediatamente antes da largada.
    work["grid_position"] = work["sess_grid_position_number"].combine_first(
        work["rd_race_grid_position_number"]
    )
    work["qualifying_position"] = work["sess_quali_position_number"].combine_first(
        work["rd_race_qualification_position_number"]
    )
    for source_column, feature_column in [
        ("sess_fp1_position_number", "fp1_position"),
        ("sess_fp2_position_number", "fp2_position"),
        ("sess_fp3_position_number", "fp3_position"),
    ]:
        if source_column in work.columns:
            work[feature_column] = work[source_column]
        else:
            work[feature_column] = pd.NA

    birth_date = pd.to_datetime(work["driver_date_of_birth"], errors="coerce")
    work["driver_age_years"] = (
        (work["race_date"] - birth_date).dt.days / 365.2425
    ).round(2)

    # A divisão é por temporada; 2025/2026 representam previsões etapa a etapa.
    work["dataset_split"] = "outside_split"
    work.loc[work["race_year"].le(2024), "dataset_split"] = "train"
    work.loc[work["race_year"].eq(2025), "dataset_split"] = "validation"
    work.loc[work["race_year"].eq(2026), "dataset_split"] = "test"

    missing_selected = sorted(
        set(FEATURE_COLUMNS + TARGET_COLUMNS + METADATA_COLUMNS) - set(work.columns)
    )
    if missing_selected:
        raise ValueError(f"Colunas necessárias ausentes para montar features: {missing_selected}")

    output_columns = list(
        dict.fromkeys(
            METADATA_COLUMNS + ["dataset_split"] + FEATURE_COLUMNS + TARGET_COLUMNS
        )
    )
    result = work[output_columns].copy()

    if len(result) != len(source):
        raise AssertionError("A etapa de features não deve remover linhas.")
    if result["target_finish_order"].isna().any():
        raise AssertionError("O alvo de ordem completa não pode ter valores ausentes.")
    if set(FEATURE_COLUMNS) & set(TARGET_COLUMNS):
        raise AssertionError("Colunas do alvo foram incluídas entre as features.")

    # As features históricas do construtor devem ser idênticas para companheiros
    # na mesma corrida, pois nenhuma delas pode depender do resultado atual.
    max_team_variation = (
        result.groupby(["rd_race_id", "rd_constructor_id"])[team_feature_columns]
        .nunique(dropna=False)
        .to_numpy()
        .max()
    )
    if max_team_variation > 1:
        raise AssertionError(
            "Histórico do construtor varia entre companheiros na mesma corrida."
        )

    return result


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH.name} não existe. Rode 02_Analise_Dados.ipynb primeiro."
        )

    source = pd.read_csv(INPUT_PATH, low_memory=False)
    _require_columns(source, REQUIRED_COLUMNS)
    source["race_date"] = pd.to_datetime(source["race_date"], errors="coerce")
    if source["race_date"].isna().any():
        raise ValueError("Há datas de corrida inválidas; não é seguro ordenar o histórico.")

    print(f"Fonte: {INPUT_PATH.name}")
    print(f"Linhas preservadas: {len(source):,}")
    print(f"Colunas limpas disponíveis: {source.shape[1]:,}")
    print(f"Corridas: {source['rd_race_id'].nunique():,}")
    print(
        "Linhas sem posição classificada: "
        f"{source['rd_position_number'].isna().sum():,}"
    )
    print("Nenhuma linha é removida nesta etapa.")

    audit = audit_source_columns(source.columns.tolist())
    print(f"\nColunas sinalizadas para não usar diretamente: {len(audit)}")
    if not audit.empty:
        summary = (
            audit.groupby("decisão")
            .agg(
                quantidade=("coluna", "size"),
                exemplos=("coluna", lambda values: ", ".join(values.head(6))),
            )
            .sort_values("quantidade", ascending=False)
        )
        print(summary.to_string())

    features = build_feature_frame(source)
    features.to_csv(OUTPUT_PATH, index=False)

    print(f"\nFeatures: {len(FEATURE_COLUMNS)}")
    print(f"Alvos: {', '.join(TARGET_COLUMNS)}")
    print(f"Linhas preservadas: {len(features):,}")
    print(f"Colunas na matriz: {features.shape[1]}")
    print(f"Arquivo salvo: {OUTPUT_PATH.name}")

    split_summary = (
        features.groupby("dataset_split", dropna=False)
        .agg(
            linhas=("rd_race_id", "size"),
            corridas=("rd_race_id", "nunique"),
            primeira_corrida=("race_date", "min"),
            ultima_corrida=("race_date", "max"),
            pilotos=("rd_driver_id", "nunique"),
        )
        .sort_index()
    )
    print("\nDivisão temporal:")
    print(split_summary.to_string())

    coverage_columns = [
        "grid_position",
        "qualifying_position",
        "fp1_position",
        "fp2_position",
        "fp3_position",
        "driver_prior_avg_finish_last5",
        "driver_circuit_prior_win_rate",
        "constructor_prior_win_rate",
    ]
    coverage = (
        features.groupby("dataset_split")[coverage_columns]
        .apply(lambda frame: frame.notna().mean().mul(100).round(1))
        .T
        .rename_axis("feature")
    )
    print("\nCobertura das features selecionadas (% não nulo):")
    print(coverage.to_string())
    print(
        "\nAmostra de 2026 é parcial no arquivo. Confira a última data antes de "
        "reportar métricas de teste."
    )


if __name__ == "__main__":
    main()
