"""Regras de leakage compartilhadas pelos notebooks de análise."""

from __future__ import annotations

import re


CURRENT_SUMMARY_PATTERN = re.compile(
    r"^(driver|constructor|engmfr|tyremfr)_(total|best)_"
)

# Campos rd_* conhecidos antes da largada e liberados explicitamente.
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
