#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_carbono_vs_aleados.py
========================
Comparación descriptiva de aceros SAE al carbono simples (10xx), resulfurados/
refosforados (11xx-12xx), alto Mn (13xx-15xx) y aleados (3xxx-9xxx).

Entradas
--------
data/processed/aceros_limpio.csv              (obligatorio; generado por 01)
data/processed/auditoria_inconsistencias.csv  (opcional)

Salidas
-------
outputs/tablas/tabla_04_*.csv                (descriptivos, contrastes, correlaciones,
                                              composición, comparabilidad, auditoría,
                                              sensibilidad a anomalías)
outputs/figuras/fig04_01 ... fig04_04        (PNG, 300 dpi por defecto)
outputs/resumen_04_carbono_vs_aleados.txt    (resumen interpretado e impreso)

Criterios metodológicos
-----------------------
* 10xx vs 3xxx-9xxx es el contraste principal; 11xx-12xx y 13xx-15xx se
  mantienen como familias independientes, NO se mezclan con 10xx.
* UTS y YS usan las columnas conciliadas de 01, salvo --valores originales.
* La comparación global entre familias es DESCRIPTIVA, no causal: composición
  y tratamiento térmico pueden estar fuertemente confundidos.
* Las comparaciones dentro de una misma condición requieren un mínimo de
  registros válidos para CADA propiedad y CADA familia.
* No se imputan cero a concentraciones de aleantes sin especificación.
* --anomalias resaltar | excluir | incluir conserva la convención de 02 y 03.
* Se exporta una sensibilidad con/sin anomalías, independientemente del modo.

Uso
---
    python src/04_carbono_vs_aleados.py
    python src/04_carbono_vs_aleados.py --anomalias excluir
    python src/04_carbono_vs_aleados.py --valores originales --min-n 5
"""

from __future__ import annotations

import argparse
import re
import textwrap
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Genera PNG sin depender de una ventana gráfica.
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy import stats


# ----------------------------------------------------------------------------
# 1. RUTAS Y CONSTANTES (independientes del directorio de ejecución)
# ----------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[1]
DIR_PROCESSED = RAIZ / "data" / "processed"
DIR_OUTPUTS = RAIZ / "outputs"
DIR_TABLAS = DIR_OUTPUTS / "tablas"
DIR_FIGURAS = DIR_OUTPUTS / "figuras"
ARCHIVO_LIMPIO = DIR_PROCESSED / "aceros_limpio.csv"
ARCHIVO_AUDITORIA = DIR_PROCESSED / "auditoria_inconsistencias.csv"
ARCHIVO_RESUMEN = DIR_OUTPUTS / "resumen_04_carbono_vs_aleados.txt"

CARBONO = "Carbono simple 10xx"
ALEADOS = "Aleados 3xxx-9xxx"
RESULF = "Resulfurados/Refosforados 11xx-12xx"
ALTO_MN = "Alto Mn 13xx-15xx"
FAMILIAS = [CARBONO, RESULF, ALTO_MN, ALEADOS]
PRINCIPALES = [CARBONO, ALEADOS]
ETIQUETAS = {
    CARBONO: "10xx",
    RESULF: "11xx-12xx",
    ALTO_MN: "13xx-15xx",
    ALEADOS: "3xxx-9xxx",
}
COLORES = {
    CARBONO: "#0072B2",
    RESULF: "#E69F00",
    ALTO_MN: "#009E73",
    ALEADOS: "#D55E00",
}
ROJO_ANOMALIA = "#D92332"

VARIABLES = {
    "UTS": ("UTS (MPa)", "MPa"),
    "YS": ("YS (MPa)", "MPa"),
    "HB": ("Dureza Brinell (HB)", "HB"),
    "ELONG": ("Elongación (%)", "%"),
    "C_pct": ("Carbono (% masa)", "%C"),
}
PROPIEDADES = ["UTS", "YS", "HB", "ELONG"]
ELEMENTOS_ALEANTES = ["Cr", "Ni", "Mo", "Mn", "Si"]
MIN_N_CORRELACION = 8
MIN_N_CONDICION = 3
RANGO_C_MINIMO = 0.15
SEMILLA_JITTER = 2026

COLUMNAS_DESCRIPTIVOS = [
    "Familia_SAE", "Condicion_Grupo", "variable", "unidad",
    "n_registros_grupo", "n", "media", "mediana", "desv_est", "min", "max",
    "n_anomalias_grupo",
]
COLUMNAS_CONTRASTE = [
    "Condicion_Grupo", "variable", "unidad", "n_10xx", "n_aleados",
    "media_10xx", "media_aleados", "diferencia_media_aleados_menos_10xx",
    "diferencia_relativa_pct", "mediana_10xx", "mediana_aleados",
    "diferencia_mediana", "nota",
]
COLUMNAS_CORRELACIONES = [
    "ambito", "Familia_SAE", "Condicion_Grupo", "variable", "unidad", "n",
    "rango_C", "pearson_r", "pearson_p", "spearman_rho", "spearman_p",
    "pendiente_por_1pctC", "pendiente_por_0_1pctC", "pendiente_ic95_inf",
    "pendiente_ic95_sup", "intercepto", "r2", "informativa", "nota",
]


# ----------------------------------------------------------------------------
# 2. ARGUMENTOS Y PREPARACIÓN DE DATOS
# ----------------------------------------------------------------------------
def leer_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comparación de familias SAE: carbono 10xx vs aleados 3xxx-9xxx."
    )
    parser.add_argument(
        "--anomalias", choices=["resaltar", "excluir", "incluir"], default="resaltar",
        help="Resaltar (predeterminado), excluir o incluir sin marcar anomalías."
    )
    parser.add_argument(
        "--valores", choices=["corregidos", "originales"], default="corregidos",
        help="Columnas de UTS/YS a utilizar (predeterminado: corregidos)."
    )
    parser.add_argument(
        "--min-n", type=int, default=MIN_N_CORRELACION,
        help=f"Mínimo orientativo para una correlación informativa (defecto: {MIN_N_CORRELACION})."
    )
    parser.add_argument(
        "--min-n-condicion", type=int, default=MIN_N_CONDICION,
        help=f"Mínimo por familia y propiedad en una misma condición (defecto: {MIN_N_CONDICION})."
    )
    parser.add_argument(
        "--rango-c-min", type=float, default=RANGO_C_MINIMO,
        help=f"Amplitud mínima de %%C para una regresión informativa (defecto: {RANGO_C_MINIMO})."
    )
    parser.add_argument("--dpi", type=int, default=300, help="Resolución de PNG (defecto: 300).")
    args = parser.parse_args()
    if args.min_n < 3 or args.min_n_condicion < 2 or args.rango_c_min < 0 or args.dpi < 72:
        parser.error("Se requiere --min-n >= 3, --min-n-condicion >= 2, "
                     "--rango-c-min >= 0 y --dpi >= 72.")
    return args


def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Lee los datos limpios y, si existe, la auditoría detallada de 01."""
    if not ARCHIVO_LIMPIO.is_file():
        raise FileNotFoundError(
            f"No existe {ARCHIVO_LIMPIO}. Ejecuta primero: python src/01_limpieza_datos.py"
        )
    df = pd.read_csv(ARCHIVO_LIMPIO, encoding="utf-8-sig", dtype={"SAE Grade": "string"})
    requeridas = [
        "registro", "SAE Grade", "Familia_SAE", "Condicion_Grupo", "C_pct",
        "UTS (MPa)", "YS (MPa)", "UTS_MPa_corr", "YS_MPa_corr",
        "Hardness_HB_num", "Elongation (%)", "flag_anomalia", "motivo_auditoria",
    ]
    faltantes = [col for col in requeridas if col not in df.columns]
    if faltantes:
        raise KeyError(f"Faltan columnas del script 01 en aceros_limpio.csv: {faltantes}")

    df["flag_anomalia"] = (
        df["flag_anomalia"].astype("string").str.strip().str.lower()
        .isin(["true", "1", "si", "sí"])
    )
    df["motivo_auditoria"] = df["motivo_auditoria"].fillna("").astype(str)
    df["Familia_SAE"] = df["Familia_SAE"].fillna("Otro").astype(str).str.strip()
    df["Condicion_Grupo"] = df["Condicion_Grupo"].fillna("Other").astype(str).str.strip()
    for col in ["C_pct", "UTS (MPa)", "YS (MPa)", "UTS_MPa_corr", "YS_MPa_corr",
                "Hardness_HB_num", "Elongation (%)"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    auditoria = None
    if ARCHIVO_AUDITORIA.is_file():
        auditoria = pd.read_csv(ARCHIVO_AUDITORIA, encoding="utf-8-sig")
    return df, auditoria


def preparar_variables(df: pd.DataFrame, usar_corregidos: bool) -> pd.DataFrame:
    """Armoniza las propiedades sin alterar valores originales del CSV."""
    d = df.copy()
    d["UTS"] = d["UTS_MPa_corr"] if usar_corregidos else d["UTS (MPa)"]
    d["YS"] = d["YS_MPa_corr"] if usar_corregidos else d["YS (MPa)"]
    d["HB"] = d["Hardness_HB_num"]   # Nunca se mezclan HRB con Brinell.
    d["ELONG"] = d["Elongation (%)"]
    return d


def familias_presentes(d: pd.DataFrame) -> list[str]:
    return [fam for fam in FAMILIAS if (d["Familia_SAE"] == fam).any()]


def formato(x, dec: int = 2) -> str:
    return "NA" if x is None or pd.isna(x) else f"{float(x):.{dec}f}"


def configurar_estilo() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "axes.grid": True, "axes.axisbelow": True,
        "grid.alpha": 0.23, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "figure.dpi": 110,
    })


def guardar_tabla(tabla: pd.DataFrame, nombre: str) -> Path:
    DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    ruta = DIR_TABLAS / nombre
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in cast")
        tabla.to_csv(ruta, index=False, encoding="utf-8-sig")
    return ruta


def guardar_figura(fig: plt.Figure, nombre: str, dpi: int) -> Path:
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    ruta = DIR_FIGURAS / nombre
    fig.savefig(ruta, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return ruta


# ----------------------------------------------------------------------------
# 3. ESTADÍSTICA DESCRIPTIVA Y CONTRASTES (SIN INFERENCIA CAUSAL)
# ----------------------------------------------------------------------------
def resumen_serie(s: pd.Series) -> dict:
    """Describe solo observaciones no faltantes."""
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "n": len(x),
        "media": float(x.mean()) if len(x) else np.nan,
        "mediana": float(x.median()) if len(x) else np.nan,
        "desv_est": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
        "min": float(x.min()) if len(x) else np.nan,
        "max": float(x.max()) if len(x) else np.nan,
    }


def tabla_descriptivos(d: pd.DataFrame, por_condicion: bool = False) -> pd.DataFrame:
    """Un renglón por (familia, [condición], variable)."""
    filas = []
    for fam in familias_presentes(d):
        subfam = d[d["Familia_SAE"] == fam]
        condiciones = sorted(subfam["Condicion_Grupo"].unique()) if por_condicion else ["Todas"]
        for cond in condiciones:
            grupo = subfam[subfam["Condicion_Grupo"] == cond] if por_condicion else subfam
            for clave, (etiqueta, unidad) in VARIABLES.items():
                filas.append({
                    "Familia_SAE": fam, "Condicion_Grupo": cond,
                    "variable": etiqueta, "unidad": unidad,
                    "n_registros_grupo": len(grupo),
                    **resumen_serie(grupo[clave]),
                    "n_anomalias_grupo": int(grupo["flag_anomalia"].sum()),
                })
    return pd.DataFrame(filas, columns=COLUMNAS_DESCRIPTIVOS)


def diagnostico_condiciones(d: pd.DataFrame, min_n_condicion: int) -> pd.DataFrame:
    """Cuenta los dos grupos principales por condición y examina soporte común."""
    filas = []
    for cond in sorted(d["Condicion_Grupo"].unique()):
        s = d[d["Condicion_Grupo"] == cond]
        a = s[s["Familia_SAE"] == CARBONO]
        b = s[s["Familia_SAE"] == ALEADOS]
        n_a, n_b = len(a), len(b)
        if n_a == 0 or n_b == 0:
            estado = "Sin representación de ambas familias"
        elif min(n_a, n_b) < min_n_condicion:
            estado = f"Muestra insuficiente en alguna familia (<{min_n_condicion})"
        else:
            estado = "Candidata (ver n válido por propiedad)"
        filas.append({
            "Condicion_Grupo": cond,
            "n_10xx": n_a, "n_aleados": n_b,
            "n_11xx_12xx": int((s["Familia_SAE"] == RESULF).sum()),
            "n_13xx_15xx": int((s["Familia_SAE"] == ALTO_MN).sum()),
            "candidata": n_a >= min_n_condicion and n_b >= min_n_condicion,
            "estado": estado,
        })
    return pd.DataFrame(filas, columns=[
        "Condicion_Grupo", "n_10xx", "n_aleados", "n_11xx_12xx",
        "n_13xx_15xx", "candidata", "estado",
    ])


def fila_contraste(a: pd.DataFrame, b: pd.DataFrame, clave: str, cond: str,
                   min_n_condicion: int) -> dict | None:
    """Diferencia aleados - 10xx; en condición específica exige tamaños mínimos."""
    x = resumen_serie(a[clave])
    y = resumen_serie(b[clave])
    umbral = 1 if cond == "Todas" else min_n_condicion
    if x["n"] < umbral or y["n"] < umbral:
        return None
    delta = y["media"] - x["media"]
    relativo = 100 * delta / x["media"] if x["media"] != 0 else np.nan
    return {
        "Condicion_Grupo": cond, "variable": VARIABLES[clave][0], "unidad": VARIABLES[clave][1],
        "n_10xx": x["n"], "n_aleados": y["n"],
        "media_10xx": x["media"], "media_aleados": y["media"],
        "diferencia_media_aleados_menos_10xx": delta,
        "diferencia_relativa_pct": relativo,
        "mediana_10xx": x["mediana"], "mediana_aleados": y["mediana"],
        "diferencia_mediana": y["mediana"] - x["mediana"],
        "nota": ("Contraste global confundido por proceso y composición; no causal"
                 if cond == "Todas" else
                 "Misma etiqueta de proceso; sin control de %C, aleantes, sección o parámetros térmicos"),
    }


def tablas_contrastes(d: pd.DataFrame, min_n_condicion: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = d[d["Familia_SAE"] == CARBONO]
    b = d[d["Familia_SAE"] == ALEADOS]
    globales, comparables = [], []
    for clave in VARIABLES:
        fila = fila_contraste(a, b, clave, "Todas", min_n_condicion)
        if fila is not None:
            globales.append(fila)
    for cond in sorted(set(a["Condicion_Grupo"]) & set(b["Condicion_Grupo"])):
        aa = a[a["Condicion_Grupo"] == cond]
        bb = b[b["Condicion_Grupo"] == cond]
        for clave in VARIABLES:
            fila = fila_contraste(aa, bb, clave, cond, min_n_condicion)
            if fila is not None:
                comparables.append(fila)
    return (pd.DataFrame(globales, columns=COLUMNAS_CONTRASTE),
            pd.DataFrame(comparables, columns=COLUMNAS_CONTRASTE))


# ----------------------------------------------------------------------------
# 4. CARBONO vs PROPIEDADES: PEARSON, SPEARMAN Y REGRESIÓN
# ----------------------------------------------------------------------------
def estadisticos_relacion(x: pd.Series, y: pd.Series,
                         min_n: int, rango_c_min: float) -> dict:
    """Cálculo descriptivo aun con n bajo; se señala si no es informativo."""
    pares = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(pares)
    rango_c = float(pares["x"].max() - pares["x"].min()) if n else np.nan
    vacio = dict.fromkeys([
        "pearson_r", "pearson_p", "spearman_rho", "spearman_p",
        "pendiente_por_1pctC", "pendiente_por_0_1pctC", "pendiente_ic95_inf",
        "pendiente_ic95_sup", "intercepto", "r2",
    ], np.nan)
    notas = []
    if n < min_n:
        notas.append(f"n={n} < {min_n}")
    if n and rango_c < rango_c_min:
        notas.append(f"rango de %C={rango_c:.3f} < {rango_c_min:.3f}")
    if n < 3:
        notas.append("se necesitan >=3 pares para regresión e IC")
    elif pares["x"].nunique() < 2:
        notas.append("%C constante: no puede ajustarse una pendiente")
    elif pares["y"].nunique() < 2:
        notas.append("propiedad constante: correlación no definida")
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pearson = stats.pearsonr(pares["x"], pares["y"])
            spearman = stats.spearmanr(pares["x"], pares["y"])
            reg = stats.linregress(pares["x"], pares["y"])
        tcrit = stats.t.ppf(0.975, df=n - 2)
        vacio = {
            "pearson_r": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
            "spearman_rho": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
            "pendiente_por_1pctC": float(reg.slope),
            "pendiente_por_0_1pctC": float(reg.slope * 0.1),
            "pendiente_ic95_inf": float(reg.slope - tcrit * reg.stderr),
            "pendiente_ic95_sup": float(reg.slope + tcrit * reg.stderr),
            "intercepto": float(reg.intercept),
            "r2": float(reg.rvalue ** 2),
        }
    informativa = bool(n >= min_n and pd.notna(rango_c) and rango_c >= rango_c_min
                       and pd.notna(vacio["pendiente_por_1pctC"]))
    return {"n": n, "rango_C": rango_c, **vacio,
            "informativa": informativa, "nota": "; ".join(notas)}


def tabla_correlaciones(d: pd.DataFrame, min_n: int, rango_c_min: float) -> pd.DataFrame:
    """Correlaciones 10xx/aleados globales y por condición (si hay >=3 pares)."""
    filas = []
    for fam in PRINCIPALES:
        grupo = d[d["Familia_SAE"] == fam]
        escenarios = [("Global", "Todas", grupo)]
        for cond in sorted(grupo["Condicion_Grupo"].unique()):
            sub = grupo[grupo["Condicion_Grupo"] == cond]
            if len(sub) >= 3:
                escenarios.append(("Dentro de condición", cond, sub))
        for ambito, cond, sub in escenarios:
            for clave in PROPIEDADES:
                e = estadisticos_relacion(sub["C_pct"], sub[clave], min_n, rango_c_min)
                filas.append({
                    "ambito": ambito, "Familia_SAE": fam, "Condicion_Grupo": cond,
                    "variable": VARIABLES[clave][0], "unidad": VARIABLES[clave][1], **e,
                })
    return pd.DataFrame(filas, columns=COLUMNAS_CORRELACIONES)


# ----------------------------------------------------------------------------
# 5. QUÍMICA NOMINAL, DIAGNÓSTICO Y SENSIBILIDAD
# ----------------------------------------------------------------------------
def tabla_composicion(d: pd.DataFrame) -> pd.DataFrame:
    """Punto medio de límites especificados; límites aislados se señalan explícitamente."""
    filas = []
    for fam in familias_presentes(d):
        s = d[d["Familia_SAE"] == fam]
        for elemento in ELEMENTOS_ALEANTES:
            col_min, col_max = f"{elemento} (Min)", f"{elemento} (Max)"
            if col_min not in s.columns or col_max not in s.columns:
                raise KeyError(f"Faltan columnas de composición: {col_min}, {col_max}")
            lo = pd.to_numeric(s[col_min], errors="coerce")
            hi = pd.to_numeric(s[col_max], errors="coerce")
            ambos = lo.notna() & hi.notna()
            solo_min = lo.notna() & hi.isna()
            solo_max = lo.isna() & hi.notna()
            nominal = pd.Series(np.nan, index=s.index, dtype=float)
            nominal.loc[ambos] = (lo.loc[ambos] + hi.loc[ambos]) / 2
            nominal.loc[solo_min] = lo.loc[solo_min]
            nominal.loc[solo_max] = hi.loc[solo_max]
            datos = resumen_serie(nominal)
            filas.append({
                "Familia_SAE": fam, "elemento": elemento, "n_total_familia": len(s),
                "n_con_dato": datos["n"], "n_rango_completo": int(ambos.sum()),
                "n_solo_min": int(solo_min.sum()), "n_solo_max": int(solo_max.sum()),
                "pct_cobertura": 100 * datos["n"] / len(s) if len(s) else np.nan,
                "media_pct": datos["media"], "mediana_pct": datos["mediana"],
                "min_pct": datos["min"], "max_pct": datos["max"],
                "nota": "Composición nominal SAE; faltante no significa 0 %; límite aislado no es punto medio",
            })
    return pd.DataFrame(filas, columns=[
        "Familia_SAE", "elemento", "n_total_familia", "n_con_dato", "n_rango_completo",
        "n_solo_min", "n_solo_max", "pct_cobertura", "media_pct", "mediana_pct",
        "min_pct", "max_pct", "nota",
    ])


def tabla_diagnostico_familias(d: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for fam in familias_presentes(d):
        s = d[d["Familia_SAE"] == fam]
        distrib = s["Condicion_Grupo"].value_counts()
        filas.append({
            "Familia_SAE": fam, "n_registros": len(s),
            "n_grados_SAE": s["SAE Grade"].nunique(),
            "n_anomalias": int(s["flag_anomalia"].sum()),
            "pct_anomalias": float(100 * s["flag_anomalia"].mean()),
            "n_condiciones": s["Condicion_Grupo"].nunique(),
            "condicion_mas_frecuente": distrib.index[0] if len(distrib) else "NA",
            "distribucion_condiciones": "; ".join(f"{g}: {n}" for g, n in distrib.items()),
            "C_pct_media": s["C_pct"].mean(),
            "C_pct_min": s["C_pct"].min(),
            "C_pct_max": s["C_pct"].max(),
        })
    return pd.DataFrame(filas, columns=[
        "Familia_SAE", "n_registros", "n_grados_SAE", "n_anomalias", "pct_anomalias",
        "n_condiciones", "condicion_mas_frecuente", "distribucion_condiciones",
        "C_pct_media", "C_pct_min", "C_pct_max",
    ])


def tabla_auditoria_familias(d: pd.DataFrame) -> pd.DataFrame:
    """Cuenta registros por tipo/categoría extraídos de motivo_auditoria del script 01."""
    filas = []
    patron = re.compile(r"\[(ANOMALIA|CORRECCION|AVISO):([A-Z0-9_]+)\]")
    for fam in familias_presentes(d):
        s = d[d["Familia_SAE"] == fam]
        conteos: dict[tuple[str, str], int] = {}
        for texto in s["motivo_auditoria"]:
            for clave in set(patron.findall(texto)):
                conteos[clave] = conteos.get(clave, 0) + 1
        for (tipo, categoria), n in sorted(conteos.items()):
            filas.append({"Familia_SAE": fam, "tipo": tipo,
                          "categoria": categoria, "n_registros": n})
    return pd.DataFrame(filas, columns=["Familia_SAE", "tipo", "categoria", "n_registros"])


def tabla_sensibilidad(d_completo: pd.DataFrame, min_n: int,
                       rango_c_min: float) -> pd.DataFrame:
    """Calcula cambios en medias y pendientes al eliminar filas con anomalías."""
    filas = []
    for fam in PRINCIPALES:
        total = d_completo[d_completo["Familia_SAE"] == fam]
        valido = total[~total["flag_anomalia"]]
        for clave in VARIABLES:
            a, b = resumen_serie(total[clave]), resumen_serie(valido[clave])
            e1 = estadisticos_relacion(total["C_pct"], total[clave], min_n, rango_c_min) if clave != "C_pct" else None
            e2 = estadisticos_relacion(valido["C_pct"], valido[clave], min_n, rango_c_min) if clave != "C_pct" else None
            filas.append({
                "Familia_SAE": fam, "variable": VARIABLES[clave][0],
                "n_todos": a["n"], "n_sin_anomalias": b["n"],
                "media_todos": a["media"], "media_sin_anomalias": b["media"],
                "delta_media_sin_menos_todos": b["media"] - a["media"],
                "pearson_r_todos": e1["pearson_r"] if e1 else np.nan,
                "pearson_r_sin_anomalias": e2["pearson_r"] if e2 else np.nan,
                "pendiente_0_1C_todos": e1["pendiente_por_0_1pctC"] if e1 else np.nan,
                "pendiente_0_1C_sin_anomalias": e2["pendiente_por_0_1pctC"] if e2 else np.nan,
            })
    return pd.DataFrame(filas, columns=[
        "Familia_SAE", "variable", "n_todos", "n_sin_anomalias",
        "media_todos", "media_sin_anomalias", "delta_media_sin_menos_todos",
        "pearson_r_todos", "pearson_r_sin_anomalias",
        "pendiente_0_1C_todos", "pendiente_0_1C_sin_anomalias",
    ])


# ----------------------------------------------------------------------------
# 6. GRÁFICAS DE PUBLICACIÓN
# ----------------------------------------------------------------------------
def pie_figura(fig: plt.Figure, mensaje: str, ancho: int = 155, y: float = 0.013) -> None:
    fig.text(0.5, y, textwrap.fill(mensaje, ancho), ha="center", va="bottom",
             fontsize=8.5, color="#3F3F3F", style="italic")


def fig_carbono_vs_propiedades(d: pd.DataFrame, corr: pd.DataFrame,
                              resaltar: bool, args: argparse.Namespace) -> Path:
    """4 paneles de dispersión; líneas solo donde el rango y n permiten interpretarlas."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=False)
    for ax, clave in zip(axes.flat, PROPIEDADES):
        sub_validos = d.dropna(subset=["C_pct", clave])
        for fam in PRINCIPALES:
            s = sub_validos[sub_validos["Familia_SAE"] == fam]
            if s.empty:
                continue
            ax.scatter(s["C_pct"], s[clave], color=COLORES[fam], alpha=0.76,
                       s=39, edgecolors="white", linewidths=0.55,
                       label=f"{ETIQUETAS[fam]} (n={len(s)})", zorder=3)
            if resaltar:
                malos = s[s["flag_anomalia"]]
                if not malos.empty:
                    ax.scatter(malos["C_pct"], malos[clave], s=132, facecolors="none",
                               edgecolors=ROJO_ANOMALIA, linewidths=1.6, zorder=4)

            r = corr[(corr["ambito"] == "Global") & (corr["Familia_SAE"] == fam)
                     & (corr["variable"] == VARIABLES[clave][0])]
            if not r.empty and bool(r.iloc[0]["informativa"]):
                info = r.iloc[0]
                xs = np.linspace(s["C_pct"].min(), s["C_pct"].max(), 80)
                ys = info["intercepto"] + info["pendiente_por_1pctC"] * xs
                ax.plot(xs, ys, color=COLORES[fam], lw=2.1, ls="--",
                        label=f"Ajuste {ETIQUETAS[fam]} (R²={info['r2']:.2f})", zorder=5)

        ax.set_title(f"{VARIABLES[clave][0]} vs carbono", fontweight="bold", pad=11)
        ax.set_xlabel("Contenido nominal de carbono, C_pct (% masa)")
        ax.set_ylabel(VARIABLES[clave][0])
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best", fontsize=8, framealpha=0.93)
        ax.margins(x=0.06, y=0.1)
    fig.suptitle("Carbono frente a propiedades mecánicas: 10xx vs aceros aleados",
                 fontsize=16, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.085, right=0.975, top=0.92, bottom=0.14,
                        wspace=0.23, hspace=0.32)
    nota = ("Líneas: regresión lineal solo con n y amplitud de %C suficientes. "
            "10xx: predominio de ferrita/perlita en condiciones no templadas; "
            "aleados: Cr, Mn, Mo y Ni modifican la templabilidad y el proceso puede producir "
            "perlita fina, bainita o martensita. No es una comparación causal; "
            "los tratamientos y el %C no están necesariamente controlados.")
    if resaltar:
        nota += " Aro rojo: registro con flag_anomalia."
    pie_figura(fig, nota, ancho=150, y=0.026)
    return guardar_figura(fig, "fig04_01_carbono_vs_propiedades_familias.png", args.dpi)


def fig_distribuciones(d: pd.DataFrame, resaltar: bool, args: argparse.Namespace) -> Path:
    """Cinco boxplots por familia; puntos individuales con jitter fijo reproducible."""
    rng = np.random.default_rng(SEMILLA_JITTER)
    familias = familias_presentes(d)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10.2))
    for ax, clave in zip(axes.flat, list(VARIABLES)):
        cajas, posiciones, grupos_validos = [], [], []
        for pos, fam in enumerate(familias, start=1):
            s = d[(d["Familia_SAE"] == fam) & d[clave].notna()]
            if s.empty:
                continue
            cajas.append(s[clave].to_numpy(dtype=float))
            posiciones.append(pos)
            grupos_validos.append((pos, fam, s))
        if cajas:
            box = ax.boxplot(cajas, positions=posiciones, widths=0.55,
                             patch_artist=True, showfliers=False,
                             medianprops={"color": "#222222", "linewidth": 2})
            for patch, (_, fam, _) in zip(box["boxes"], grupos_validos):
                patch.set_facecolor(COLORES[fam])
                patch.set_alpha(0.37)
                patch.set_edgecolor(COLORES[fam])
            for pos, fam, s in grupos_validos:
                x = pos + rng.uniform(-0.17, 0.17, size=len(s))
                ax.scatter(x, s[clave], s=16, alpha=0.67,
                           c=COLORES[fam], linewidths=0, zorder=3)
                if resaltar:
                    m = s["flag_anomalia"].to_numpy(dtype=bool)
                    if m.any():
                        ax.scatter(x[m], s[clave].to_numpy()[m], facecolors="none",
                                   edgecolors=ROJO_ANOMALIA, s=94, linewidths=1.4, zorder=4)
        ax.set_title(VARIABLES[clave][0], fontweight="bold", pad=9)
        ax.set_ylabel(VARIABLES[clave][1])
        ax.set_xticks(range(1, len(familias) + 1))
        ax.set_xticklabels([f"{ETIQUETAS[f]}\n(n={(d['Familia_SAE'] == f).sum()})"
                            for f in familias], fontsize=8)
        ax.grid(axis="x", visible=False)
    # Sexto panel: claves interpretativas, no datos inventados.
    ax_info = axes.flat[5]
    ax_info.axis("off")
    ax_info.text(0.02, 0.98, "Lectura metalúrgica", transform=ax_info.transAxes,
                 va="top", fontweight="bold", fontsize=12)
    for i, linea in enumerate([
        "10xx: ferrita y perlita habituales; mayor %C",
        "favorece más perlita y mayor dureza/resistencia.",
        "Aleados: elementos de aleación y velocidad de",
        "enfriamiento modifican la microestructura.",
        "Las distribuciones mezclan condiciones SAE:",
        "no equivalen al efecto aislado de alear.",
    ]):
        ax_info.text(0.02, 0.87 - i * 0.115, linea, transform=ax_info.transAxes,
                     va="top", fontsize=10)
    fig.suptitle("Distribución de propiedades por familia SAE",
                 fontsize=16, fontweight="bold", y=0.982)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.91, bottom=0.13,
                        wspace=0.28, hspace=0.32)
    nota = ("La línea negra de cada caja es la mediana; cajas = rango intercuartílico. "
            "Puntos = registros nominales SAE (no probetas independientes). "
            "Se mantienen separados 11xx-12xx y 13xx-15xx; no se agregan a 10xx. "
            "Los tamaños se indican por familia y pueden diferir por propiedad (faltantes).")
    if resaltar:
        nota += " Aros rojos = registros auditados con anomalía."
    pie_figura(fig, nota, ancho=150, y=0.019)
    return guardar_figura(fig, "fig04_02_distribuciones_por_familia.png", args.dpi)


def fig_composicion(comp: pd.DataFrame, args: argparse.Namespace) -> Path:
    """Mapa térmico de aleantes nominales; ausente = sin especificación, nunca cero."""
    familias = [f for f in FAMILIAS if (comp["Familia_SAE"] == f).any()]
    matriz = np.full((len(familias), len(ELEMENTOS_ALEANTES)), np.nan)
    etiquetas = {}
    for i, fam in enumerate(familias):
        for j, elemento in enumerate(ELEMENTOS_ALEANTES):
            r = comp[(comp["Familia_SAE"] == fam) & (comp["elemento"] == elemento)]
            if not r.empty and int(r.iloc[0]["n_con_dato"]) > 0:
                matriz[i, j] = float(r.iloc[0]["mediana_pct"])
                etiquetas[(i, j)] = (int(r.iloc[0]["n_con_dato"]),
                                    int(r.iloc[0]["n_total_familia"]))
    datos_m = np.ma.masked_invalid(matriz)
    maximo_composicion = float(np.nanmax(matriz)) if np.isfinite(matriz).any() else 0.0
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#E7E7E7")
    fig, ax = plt.subplots(figsize=(11, max(5.4, 0.9 * len(familias) + 2.3)))
    imagen = ax.imshow(datos_m, cmap=cmap, aspect="auto", vmin=0)
    ax.set_xticks(range(len(ELEMENTOS_ALEANTES)), ELEMENTOS_ALEANTES)
    ax.set_yticks(range(len(familias)), [ETIQUETAS[f] for f in familias])
    ax.tick_params(length=0)
    ax.grid(False)
    for i in range(len(familias)):
        for j in range(len(ELEMENTOS_ALEANTES)):
            if (i, j) not in etiquetas:
                texto = "Sin dato"
            else:
                n, total = etiquetas[(i, j)]
                texto = f"{matriz[i, j]:.2f} %\n(n={n}/{total})"
            ax.text(j, i, texto, va="center", ha="center", fontsize=9,
                    color="white" if np.isfinite(matriz[i, j]) and matriz[i, j] >
                    0.62 * maximo_composicion else "#202020")
    cb = fig.colorbar(imagen, ax=ax, fraction=0.043, pad=0.03)
    cb.set_label("Mediana de composición nominal (% masa)")
    ax.set_title("Elementos aleantes por familia SAE: Cr, Ni, Mo, Mn y Si",
                 fontsize=14, fontweight="bold", pad=15)
    fig.subplots_adjust(left=0.15, right=0.94, top=0.84, bottom=0.23)
    pie_figura(fig,
        "Cada celda muestra mediana del punto medio (Min-Max); si hay un solo límite, se usa "
        "como dato nominal provisional. n = registros que informan el elemento / total de familia. "
        "Sin dato NO significa 0 %. El mayor contenido nominal no prueba una fase, un carburo "
        "ni un tratamiento específico: hacen falta historia térmica y caracterización.",
        ancho=133, y=0.02)
    return guardar_figura(fig, "fig04_03_composicion_nominal_aleantes.png", args.dpi)


def fig_disponibilidad_condiciones(diag: pd.DataFrame, args: argparse.Namespace) -> Path:
    """Muestra dónde existen verdaderos subconjuntos comparables por proceso."""
    orden = diag.sort_values(["n_10xx", "n_aleados"], ascending=False)
    n = len(orden)
    fig, ax = plt.subplots(figsize=(12.6, max(5.2, 0.58 * n + 2.8)))
    y = np.arange(n)
    ax.barh(y - 0.19, orden["n_10xx"], height=0.35,
            label="Carbono 10xx", color=COLORES[CARBONO])
    ax.barh(y + 0.19, orden["n_aleados"], height=0.35,
            label="Aleados 3xxx-9xxx", color=COLORES[ALEADOS])
    ax.set_yticks(y, orden["Condicion_Grupo"])
    ax.invert_yaxis()
    ax.set_xlabel("Registros por familia y condición de procesamiento")
    ax.set_title("Soporte común para comparar 10xx frente a aleados dentro de un proceso",
                 fontsize=14, fontweight="bold", pad=15)
    ax.axvline(args.min_n_condicion, color="#555555", ls="--", lw=1.1,
               label=f"Mínimo orientativo: n={args.min_n_condicion}")
    ax.set_xlim(0, max(3, int(max(orden["n_10xx"].max(),
                                 orden["n_aleados"].max())) * 1.30))
    ax.grid(axis="y", visible=False)
    # La marca de comparabilidad se coloca en coordenadas de datos para
    # mantener la alineación incluso cuando se invierte el eje vertical.
    extremo_x = ax.get_xlim()[1]
    for yi, (_, r) in enumerate(orden.iterrows()):
        if bool(r["candidata"]):
            ax.text(extremo_x * 0.985, yi, "✓ comparable", ha="right",
                    va="center", fontsize=8.1, color="#27684F")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=9, frameon=False)
    fig.subplots_adjust(left=0.22, right=0.73, top=0.87, bottom=0.19)
    pie_figura(fig,
        "La misma etiqueta de proceso no asegura igual temperatura, tiempo, sección, velocidad "
        "de enfriamiento ni %C. Condiciones sin n suficiente en AMBAS familias no se usan para "
        "contrastes intraproceso; n por propiedad también puede caer por valores faltantes.",
        ancho=140, y=0.023)
    return guardar_figura(fig, "fig04_04_disponibilidad_condiciones.png", args.dpi)


# ----------------------------------------------------------------------------
# 7. RESUMEN TÉCNICO: DINÁMICO Y EXPLÍCITO SOBRE LIMITACIONES
# ----------------------------------------------------------------------------
def construir_resumen(args: argparse.Namespace, completo: pd.DataFrame,
                     usado: pd.DataFrame, auditoria: pd.DataFrame | None,
                     descriptivos: pd.DataFrame, contraste: pd.DataFrame,
                     intra: pd.DataFrame, correlaciones: pd.DataFrame,
                     diagnostico: pd.DataFrame, conds: pd.DataFrame,
                     composicion: pd.DataFrame, audit_fam: pd.DataFrame,
                     sensibilidad: pd.DataFrame, rutas: list[Path]) -> str:
    L: list[str] = []
    sep = "=" * 91

    def encabezado(titulo: str) -> None:
        L.extend(["", sep, titulo, sep])

    def parrafo(texto: str) -> None:
        L.append(textwrap.fill(texto, width=89, initial_indent="  ", subsequent_indent="  "))

    L.extend([sep, "RESUMEN 04 · COMPARACIÓN SAE 10xx FRENTE A 3xxx-9xxx", sep])
    L.append(f"Origen: data/processed/aceros_limpio.csv | resistencia: {args.valores} | anomalías: {args.anomalias}")
    L.append(f"Total base: {len(completo)} | familias consideradas: "
             f"{int(completo['Familia_SAE'].isin(FAMILIAS).sum())} | "
             f"marcados (base): {int(completo['flag_anomalia'].sum())} | "
             f"registros utilizados: {len(usado)}")
    if auditoria is None:
        L.append("Auditoría detallada: archivo opcional no encontrado; se usan flags y motivos del CSV limpio.")
    else:
        L.append(f"Auditoría detallada: {len(auditoria)} renglones de hallazgos (incluye avisos y correcciones).")
    if args.valores == "originales":
        L.append("ADVERTENCIA: UTS/YS originales pueden contener discrepancias de escala y captura.")
    L.append("La bandera de anomalía es por registro, no por propiedad: excluir elimina la fila completa.")

    encabezado("1. COBERTURA Y DIFERENCIAS DE PROCESAMIENTO")
    for _, r in diagnostico.iterrows():
        L.append(f"  {ETIQUETAS[r['Familia_SAE']]:<12} n={int(r['n_registros']):>3}, "
                 f"grados={int(r['n_grados_SAE']):>3}, "
                 f"flag={int(r['n_anomalias']):>2}, "
                 f"C medio={formato(r['C_pct_media'], 3)} %, "
                 f"rango C={formato(r['C_pct_min'])}-{formato(r['C_pct_max'])} %")
        parrafo(f"Condiciones: {r['distribucion_condiciones']}")
    n_comp = int(conds["candidata"].sum()) if not conds.empty else 0
    L.append(f"  Condiciones candidatas con ambas familias (>= {args.min_n_condicion} filas): {n_comp}.")
    L.append(f"  Comparaciones intraproceso realmente evaluables por variable: {len(intra)}.")
    if intra.empty:
        parrafo("No se han encontrado comparaciones 10xx vs aleados con tamaños suficientes "
                "dentro de la MISMA condición. El contraste global refleja grupos de procesos distintos; "
                "no atribuir diferencias medias a los elementos de aleación.")
    else:
        for cond, s in intra.groupby("Condicion_Grupo", sort=True):
            L.append(f"  - {cond}: {len(s)} variables con contraste descriptivo.")

    encabezado("2. ESTADÍSTICOS Y CONTRASTE PRINCIPAL (ALEADOS - 10xx)")
    for fam in PRINCIPALES:
        s = descriptivos[(descriptivos["Familia_SAE"] == fam)]
        if s.empty:
            L.append(f"  {ETIQUETAS[fam]}: sin registros.")
            continue
        L.append(f"  {ETIQUETAS[fam]}:")
        for _, r in s.iterrows():
            L.append(f"    {r['variable']:<23} n={int(r['n']):>3} "
                     f"media={formato(r['media']):>8} "
                     f"mediana={formato(r['mediana']):>8} "
                     f"DE={formato(r['desv_est']):>7} "
                     f"mín={formato(r['min']):>8} máx={formato(r['max']):>8}")
    if contraste.empty:
        L.append("  No existe muestra de ambas familias para calcular diferencias de medias.")
    else:
        L.append("  Diferencias de medias (aleados - 10xx), SOLO descriptivas:")
        for _, r in contraste.iterrows():
            L.append(f"    {r['variable']:<23} "
                     f"Δ={formato(r['diferencia_media_aleados_menos_10xx']):>8} "
                     f"{r['unidad']:<4} ({formato(r['diferencia_relativa_pct'], 1)} % respecto a 10xx)")
    parrafo("Un promedio más alto en aleados no demuestra fortalecimiento por aleación. "
            "Los grupos pueden representar distintos porcentajes de C, grados, tratamientos "
            "(normalizado, recocido, estirado, temple/revenido) y condiciones de enfriamiento.")

    encabezado("3. SENSIBILIDAD AL CARBONO POR FAMILIA")
    L.append(f"  Correlaciones informativas: n >= {args.min_n} y rango C >= {args.rango_c_min:.2f} %.")
    L.append("  Pendientes expresadas por incremento de +0.10 puntos porcentuales de carbono.")
    for fam in PRINCIPALES:
        L.append(f"  {ETIQUETAS[fam]}:")
        s = correlaciones[(correlaciones["Familia_SAE"] == fam) &
                          (correlaciones["ambito"] == "Global")]
        for _, r in s.iterrows():
            marca = "INTERPRETABLE CON RESERVAS" if r["informativa"] else "NO INFORMATIVA"
            L.append(f"    {r['variable']:<23} n={int(r['n']):>3} "
                     f"r={formato(r['pearson_r'], 3):>6} "
                     f"p={formato(r['pearson_p'], 3):>6} "
                     f"rho={formato(r['spearman_rho'], 3):>6} "
                     f"pendiente={formato(r['pendiente_por_0_1pctC']):>8} "
                     f"{r['unidad']}/0.1%C; R²={formato(r['r2'], 3)}  [{marca}]")
            if r["nota"]:
                L.append(f"      Limitación: {r['nota']}")
    parrafo("La pendiente global de un acero aleado NO mide aisladamente el efecto del C: "
            "Cr, Ni, Mo, Mn, Si, el tratamiento térmico y su interacción cambian a la vez. "
            "Las correlaciones por condición se exportan para controlar parcialmente el proceso.")

    encabezado("4. INTERPRETACIÓN METALÚRGICA")
    for texto in [
        "Aceros simples 10xx: su respuesta habitual en estados no templados depende principalmente "
        "del porcentaje de carbono y del enfriamiento. En el campo hipoeutectoide aumentos de %C "
        "elevan la fracción de perlita frente a ferrita proeutectoide; generalmente suben UTS, YS "
        "y HB y desciende la elongación. La cinética, el tamaño de grano y la deformación previa "
        "también influyen; un 10xx templado puede formar martensita.",
        "Aceros aleados 3xxx-9xxx: elementos como Mn, Cr, Mo y Ni pueden retrasar las "
        "transformaciones difusionales de austenita a ferrita/perlita y desplazar las curvas TTT/CCT. "
        "Por ello aumenta la templabilidad: según composición, espesor y enfriamiento, se favorecen "
        "perlita más fina, bainita o martensita. Templabilidad NO es sinónimo de dureza máxima.",
        "Fortalecimiento adicional: Mn, Ni y Si pueden contribuir por solución sólida; Cr y Mo "
        "pueden participar en la formación de carburos aleados y endurecimiento por precipitación "
        "en ciclos térmicos adecuados. Si también influye en desoxidación y estabilidad de carburos. "
        "La presencia nominal del elemento NO comprueba que existan precipitados específicos.",
        "Grupos intermedios: los 11xx-12xx incorporan prácticas de composición orientadas "
        "a maquinabilidad (S/P, según grado), y 13xx-15xx presentan Mn elevado. No se deben "
        "sumar automáticamente a 10xx: difieren en química y comportamiento de procesamiento.",
        "La microestructura final no está observada en el CSV. Sin metalografía, historia térmica, "
        "dimensiones o curvas de enfriamiento, las explicaciones anteriores son mecanismos "
        "metalúrgicos plausibles, NO fases verificadas experimentalmente en cada registro.",
    ]:
        parrafo(texto)

    encabezado("5. DATOS NOMINALES DE Cr, Ni, Mo, Mn Y Si")
    for fam in PRINCIPALES:
        L.append(f"  {ETIQUETAS[fam]} (medianas de datos disponibles; % masa):")
        s = composicion[composicion["Familia_SAE"] == fam]
        for _, r in s.iterrows():
            L.append(f"    {r['elemento']:<3} = {formato(r['mediana_pct'], 3):>7} % "
                     f"(cobertura {int(r['n_con_dato'])}/{int(r['n_total_familia'])}; "
                     f"rangos completos {int(r['n_rango_completo'])})")
    parrafo("Estos valores representan composiciones nominales de tabla, no química medida por colada. "
            "Cuando solo aparece Min o Max se usa ese límite como valor provisional; "
            "un elemento sin dato NO equivale a 0 %. No es adecuado inferir fases o precipitados "
            "a partir de la tabla de composición solamente.")

    encabezado("6. ANOMALÍAS, SENSIBILIDAD Y LIMITACIONES")
    for fam in PRINCIPALES:
        s = sensibilidad[sensibilidad["Familia_SAE"] == fam]
        L.append(f"  {ETIQUETAS[fam]}: comparación 'todos' vs 'sin flag_anomalia'")
        for _, r in s.iterrows():
            L.append(f"    {r['variable']:<23} n={int(r['n_todos'])} -> {int(r['n_sin_anomalias'])}, "
                     f"Δ media={formato(r['delta_media_sin_menos_todos'])}")
    incidencias = audit_fam[audit_fam["tipo"] == "ANOMALIA"].sort_values(
        "n_registros", ascending=False).head(10)
    if incidencias.empty:
        L.append("  Sin categorías ANOMALIA documentadas en motivo_auditoria para las familias incluidas.")
    else:
        L.append("  Principales categorías de anomalía (n de registros afectados):")
        for _, r in incidencias.iterrows():
            L.append(f"    {ETIQUETAS[r['Familia_SAE']]:<12} {r['categoria']:<27} {int(r['n_registros']):>3}")
    for texto in [
        "El número de filas no equivale al número de probetas ensayadas. Los valores son nominales "
        "y hay posible dependencia entre grados cercanos; por tanto, los p-valores de correlación "
        "son exploratorios, no evidencia experimental independiente.",
        "Las familias tienen tamaños muestrales posiblemente desiguales. Los faltantes por "
        "propiedad reducen n y los datos de dureza HRB no se convierten artificialmente a HB.",
        "El rango de carbono C_pct puede proceder del promedio de Min/Max o solo de un límite "
        "y NO es necesariamente el contenido real de una colada.",
        "La comparación por condición reduce parte de la confusión, pero no controla sección, "
        "temperaturas exactas, velocidad de enfriamiento, deformación, contenido preciso de aleantes "
        "ni interacciones entre estos factores. No se atribuye causalidad ni se aplica ANOVA a "
        "grupos SAE potencialmente no intercambiables.",
        "La sensibilidad al excluir anomalías es conservadora: flag_anomalia puede deberse "
        "a una sola columna y elimina la fila para todas las variables. Las correcciones del "
        "script 01 no deben interpretarse como mediciones verificadas.",
    ]:
        parrafo(texto)

    encabezado("7. ARCHIVOS GENERADOS")
    for ruta in rutas:
        L.append(f"  - {ruta.relative_to(RAIZ)}")
    return "\n".join(L)


# ----------------------------------------------------------------------------
# 8. PROGRAMA PRINCIPAL
# ----------------------------------------------------------------------------
def main() -> None:
    args = leer_argumentos()
    configurar_estilo()
    df_raw, auditoria = cargar_datos()
    completo = preparar_variables(df_raw, usar_corregidos=(args.valores == "corregidos"))
    completo = completo[completo["Familia_SAE"].isin(FAMILIAS)].copy()
    if completo.empty:
        raise SystemExit("El CSV no contiene familias SAE 10xx, 11xx-12xx, 13xx-15xx o 3xxx-9xxx.")

    usado = (completo.loc[~completo["flag_anomalia"]].copy()
             if args.anomalias == "excluir" else completo.copy())
    if usado.empty:
        raise SystemExit("No quedan registros tras excluir anomalías. Use --anomalias resaltar o incluir.")
    resaltar = args.anomalias == "resaltar"

    # Tablas y diagnósticos; nunca se imputan propiedades mecánicas faltantes.
    descriptivos = tabla_descriptivos(usado)
    por_condicion = tabla_descriptivos(usado, por_condicion=True)
    conds = diagnostico_condiciones(usado, args.min_n_condicion)
    contraste, intra = tablas_contrastes(usado, args.min_n_condicion)
    correlaciones = tabla_correlaciones(usado, args.min_n, args.rango_c_min)
    composicion = tabla_composicion(usado)
    diagnostico = tabla_diagnostico_familias(usado)
    audit_fam = tabla_auditoria_familias(completo)  # Auditoría incluye filas excluidas.
    sensibilidad = tabla_sensibilidad(completo, args.min_n, args.rango_c_min)

    rutas_tablas = [
        guardar_tabla(descriptivos, "tabla_04_descriptivos_por_familia.csv"),
        guardar_tabla(por_condicion, "tabla_04_descriptivos_familia_condicion.csv"),
        guardar_tabla(contraste, "tabla_04_contraste_global_10xx_vs_aleados.csv"),
        guardar_tabla(intra, "tabla_04_contrastes_condiciones_comparables.csv"),
        guardar_tabla(conds, "tabla_04_diagnostico_condiciones.csv"),
        guardar_tabla(correlaciones, "tabla_04_correlaciones_c_vs_propiedades.csv"),
        guardar_tabla(composicion, "tabla_04_composicion_aleantes.csv"),
        guardar_tabla(diagnostico, "tabla_04_diagnostico_familias.csv"),
        guardar_tabla(audit_fam, "tabla_04_auditoria_por_familia.csv"),
        guardar_tabla(sensibilidad, "tabla_04_sensibilidad_anomalias.csv"),
    ]
    rutas_figuras = [
        fig_carbono_vs_propiedades(usado, correlaciones, resaltar, args),
        fig_distribuciones(usado, resaltar, args),
        fig_composicion(composicion, args),
        fig_disponibilidad_condiciones(conds, args),
    ]

    DIR_OUTPUTS.mkdir(parents=True, exist_ok=True)
    resumen = construir_resumen(
        args, completo, usado, auditoria, descriptivos, contraste, intra,
        correlaciones, diagnostico, conds, composicion, audit_fam, sensibilidad,
        rutas_tablas + rutas_figuras + [ARCHIVO_RESUMEN],
    )
    ARCHIVO_RESUMEN.write_text(resumen + "\n", encoding="utf-8")
    print(resumen)


if __name__ == "__main__":
    main()
