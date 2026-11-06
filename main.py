#!/usr/bin/env python3
"""Processa yellow_tripdata_2016-02.csv em paralelo e valida o resultado."""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


NUMERIC_COLUMNS = (
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "total_amount",
)
DATE_COLUMN = "tpep_pickup_datetime"
PAYMENT_COLUMN = "payment_type"
CSV_COLUMNS = [DATE_COLUMN, PAYMENT_COLUMN, *NUMERIC_COLUMNS]
MISSING = "(missing)"
REL_TOLERANCE = 1e-9
ABS_TOLERANCE = 1e-6


def map_chunk(frame: pd.DataFrame) -> dict[str, Any]:
    """Resume um bloco; nunca devolve o DataFrame ao processo principal."""
    partial: dict[str, Any] = {
        "rows_processed": int(len(frame)),
        "numeric": {},
        "pickup_date_counts": {},
        "payment_type_counts": {},
    }

    pickup_dates = frame[DATE_COLUMN].astype("string").str.slice(0, 10)
    valid_dates = pd.to_datetime(pickup_dates, format="%Y-%m-%d", errors="coerce")
    pickup_dates = pickup_dates.where(valid_dates.notna(), MISSING).fillna(MISSING)
    partial["pickup_date_counts"] = {
        str(key): int(value)
        for key, value in pickup_dates.value_counts().items()
    }

    payments = frame[PAYMENT_COLUMN].astype("string").fillna(MISSING).str.strip()
    payments = payments.replace("", MISSING)
    partial["payment_type_counts"] = {
        str(key): int(value) for key, value in payments.value_counts().items()
    }

    for column in NUMERIC_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        partial["numeric"][column] = {
            "count": int(values.size),
            "sum": float(values.sum()),
            "min": float(values.min()) if not values.empty else None,
            "max": float(values.max()) if not values.empty else None,
        }
    return partial


def reduce_partials(partials: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Combina somas, contagens, extremos e frequências dos blocos."""
    result: dict[str, Any] = {
        "rows_processed": 0,
        "numeric": {
            column: {"count": 0, "sum": 0.0, "min": None, "max": None}
            for column in NUMERIC_COLUMNS
        },
        "pickup_date_counts": {},
        "payment_type_counts": {},
    }
    for partial in partials:
        result["rows_processed"] += partial["rows_processed"]
        for column in NUMERIC_COLUMNS:
            target = result["numeric"][column]
            source = partial["numeric"][column]
            target["count"] += source["count"]
            target["sum"] += source["sum"]
            if source["min"] is not None:
                target["min"] = source["min"] if target["min"] is None else min(target["min"], source["min"])
                target["max"] = source["max"] if target["max"] is None else max(target["max"], source["max"])
        for key in ("pickup_date_counts", "payment_type_counts"):
            for label, count in partial[key].items():
                result[key][label] = result[key].get(label, 0) + count

    for values in result["numeric"].values():
        values["mean"] = values["sum"] / values["count"] if values["count"] else None
    result["pickup_date_counts"] = dict(sorted(result["pickup_date_counts"].items()))
    result["payment_type_counts"] = dict(sorted(result["payment_type_counts"].items()))
    return result


def read_chunks(path: Path, chunksize: int):
    return pd.read_csv(
        path,
        usecols=CSV_COLUMNS,
        chunksize=chunksize,
        low_memory=False,
    )


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(data, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")


def metrics_from_result(result: Any) -> Any:
    if isinstance(result, dict) and "metrics" in result:
        return result["metrics"]
    return result


def compare_values(expected: Any, actual: Any, location: str = "metrics") -> list[str]:
    errors: list[str] = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            errors.append(f"{location}: chaves diferentes")
            return errors
        for key in expected:
            errors.extend(compare_values(expected[key], actual[key], f"{location}.{key}"))
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            errors.append(f"{location}: esperado número, recebido {type(actual).__name__}")
        elif not math.isclose(
            float(expected), float(actual), rel_tol=REL_TOLERANCE, abs_tol=ABS_TOLERANCE
        ):
            errors.append(f"{location}: {expected} != {actual}")
    elif expected != actual:
        errors.append(f"{location}: {expected!r} != {actual!r}")
    return errors


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8", mode="w"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("o valor deve ser maior que zero")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Processamento paralelo de corridas de táxi com divisão e conquista."
    )
    parser.add_argument(
        "--arquivo",
        type=Path,
        default=Path(__file__).resolve().parent / "yellow_tripdata_2016-02.csv",
        help="caminho do CSV de entrada",
    )
    parser.add_argument(
        "--processos",
        type=positive_int,
        default=os.cpu_count() or 1,
        help="número de processos (padrão: núcleos lógicos disponíveis)",
    )
    parser.add_argument(
        "--chunksize",
        type=positive_int,
        default=100_000,
        help="linhas por bloco CSV (padrão: 100000)",
    )
    parser.add_argument(
        "--diretorio-resultados",
        type=Path,
        default=Path(__file__).resolve().parent / "resultados",
        help="pasta para JSON e log",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = args.arquivo.resolve()
    output_dir = args.diretorio_resultados.resolve()
    result_path = output_dir / "resultados_parte_b.json"
    reference_path = output_dir / "resultados_parte_a.json"
    log_path = output_dir / "execucao_parte_b.log"
    configure_logging(log_path)

    if not csv_path.is_file():
        logging.error("Arquivo CSV não encontrado: %s", csv_path)
        return 2

    logging.info("Início do processamento distribuído (multiprocessing).")
    logging.info("Arquivo: %s (%s bytes)", csv_path, csv_path.stat().st_size)
    logging.info("Núcleos lógicos disponíveis: %s", os.cpu_count() or "indisponível")
    logging.info("Processos: %d | chunksize CSV: %d linhas", args.processos, args.chunksize)
    logging.info("Estratégia: leitura em blocos, map por worker e reduce final.")

    if reference_path.exists():
        logging.info("Referência Parte A encontrada: %s", reference_path)
    else:
        logging.info(
            "Referência Parte A ausente; gerando resultados_parte_a.json "
            "com execução sequencial da mesma função map/reduce."
        )
        sequential_start = time.perf_counter()
        sequential_metrics = reduce_partials(
            map_chunk(chunk) for chunk in read_chunks(csv_path, args.chunksize)
        )
        sequential_seconds = time.perf_counter() - sequential_start
        write_json(
            reference_path,
            {
                "version": 1,
                "execution": "sequential_reference",
                "elapsed_seconds": sequential_seconds,
                "metrics": sequential_metrics,
            },
        )
        logging.info(
            "Referência sequencial concluída em %.3f s; %d linhas processadas.",
            sequential_seconds,
            sequential_metrics["rows_processed"],
        )

    try:
        with reference_path.open("r", encoding="utf-8") as source:
            reference = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        logging.error("Não foi possível ler a referência %s: %s", reference_path, error)
        return 2

    parallel_start = time.perf_counter()
    try:
        with mp.Pool(processes=args.processos) as pool:
            partials = pool.imap_unordered(
                map_chunk, read_chunks(csv_path, args.chunksize), chunksize=1
            )
            metrics = reduce_partials(partials)
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as error:
        logging.exception("Falha durante o processamento paralelo: %s", error)
        return 1
    parallel_seconds = time.perf_counter() - parallel_start

    errors = compare_values(metrics_from_result(reference), metrics)
    validation = {
        "passed": not errors,
        "reference": str(reference_path),
        "rel_tolerance": REL_TOLERANCE,
        "abs_tolerance": ABS_TOLERANCE,
        "errors": errors,
    }
    result = {
        "version": 1,
        "execution": "parallel",
        "input_file": csv_path.name,
        "processes": args.processos,
        "cpu_count": os.cpu_count() or 1,
        "chunksize_rows": args.chunksize,
        "elapsed_seconds": parallel_seconds,
        "metrics": metrics,
        "validation": validation,
    }
    write_json(result_path, result)

    logging.info("Processamento paralelo concluído em %.3f s.", parallel_seconds)
    logging.info("Total de linhas processadas: %d", metrics["rows_processed"])
    logging.info(
        "Validação automática contra resultados_parte_a.json: %s "
        "(tolerâncias rel=%g, abs=%g).",
        "APROVADA" if validation["passed"] else "FALHOU",
        REL_TOLERANCE,
        ABS_TOLERANCE,
    )
    if errors:
        for error in errors[:20]:
            logging.error("Divergência: %s", error)
        logging.error("Total de divergências: %d", len(errors))
    logging.info("Resultado: %s", result_path)
    logging.info("Log: %s", log_path)
    return 0 if validation["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())