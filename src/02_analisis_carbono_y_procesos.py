#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_analisis_carbono_y_procesos.py
=================================
Análisis de (1) %C vs propiedades mecánicas y (2) comparación entre condiciones de
procesamiento, conectando los resultados con la metalurgia y la microestructura.

Entradas
--------
data/processed/aceros_limpio.csv                 (obligatoria, generada por 01)
data/processed/auditoria_inconsistencias.csv     (opcional, solo para el resumen)

Salidas
-------
data/processed/tabla_02_correlaciones_c_vs_propiedades.csv   Pearson/Spearman/regresión por subgrupo
data/processed/tabla_02_estadisticas_por_condicion.csv       n, media, mediana, desv. est., mín, máx
data/processed/tabla_02_pares_hot_rolled_vs_cold_drawn.csv   Δ por SAE Grade (composición idéntica)
data/processed/tabla_02_resumen_pares_hot_rolled_vs_cold_drawn.csv
data/processed/tabla_02_sensibilidad_anomalias.csv           resultados con y sin registros marcados
outputs/resumen_02_carbono_procesos.txt                      resumen interpretado (también se imprime)
outputs/figuras/fig02_01 ... fig02_06 (PNG, 300 dpi)

Manejo de anomalías (flag_anomalia, generada por 01)
----------------------------------------------------
  --anomalias resaltar (defecto) : se incluyen y se marcan con un aro rojo en las gráficas.
  --anomalias excluir            : se excluyen de tablas y gráficas.
  --anomalias incluir            : se incluyen sin resaltar.
En cualquier modo se exporta la tabla de sensibilidad (con vs sin anomalías).

Valores de resistencia
----------------------
  --valores corregidos (defecto) : usa UTS_MPa_corr / YS_MPa_corr (conciliados con ksi en el script 01).
  --valores originales           : usa UTS (MPa) / YS (MPa) tal como vienen en la fuente.

Uso
---
    python src/02_analisis_carbono_y_procesos.py
    python src/02_analisis_carbono_y_procesos.py --anomalias excluir
"""

from __future__ import annotations

import argparse
import re
import textwrap
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: funciona en servidores/CI y en VS Code
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

# ----------------------------------------------------------------------------
# 1. RUTAS
# ----------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[1]
DIR_PROCESSED = RAIZ / "data" / "processed"
DIR_OUTPUTS = RAIZ / "outputs"
DIR_FIGURAS = DIR_OUTPUTS / "figuras"

ARCHIVO_LIMPIO = DIR_PROCESSED / "aceros_limpio.csv"
ARCHIVO_AUDITORIA = DIR_PROCESSED / "auditoria_inconsistencias.csv"

# ----------------------------------------------------------------------------
# 2. CONFIGURACIÓN
# ----------------------------------------------------------------------------
# Variables de análisis: clave interna -> etiqueta con unidades
VARIABLES = {
    "UTS": "UTS (MPa)",
    "YS": "YS (MPa)",
    "HB": "Dureza Brinell (HB)",
    "ELONG": "Elongación (%)",
}
UNIDAD_CORTA = {"UTS": "MPa", "YS": "MPa", "HB": "HB", "ELONG": "p.p."}  # p.p. = puntos porcentuales
SIGNO_ESPERADO = {"UTS": +1, "YS": +1, "HB": +1, "ELONG": -1}

TITULO_PANEL = {
    "UTS": "(a) UTS vs %C\n↑ perlita / Fe3C ⇒ ↑ resistencia a la tracción",
    "YS": "(b) YS vs %C\n↑ obstáculos al deslizamiento de dislocaciones ⇒ ↑ fluencia",
    "HB": "(c) Dureza vs %C\n↑ fracción de perlita (cementita) ⇒ ↑ dureza",
    "ELONG": "(d) Elongación vs %C\n↓ ferrita proeutectoide dúctil ⇒ ↓ ductilidad",
}

ORDEN_CONDICIONES = [
    "Hot rolled", "Cold drawn", "Annealed", "Annealed + Cold drawn",
    "Spheroidized annealed", "Normalized", "Normalized + Cold drawn", "Quenched & tempered",
]
COLORES_CONDICION = {  # paleta apta para daltonismo (Okabe-Ito ampliada)
    "Hot rolled": "#E69F00",
    "Cold drawn": "#0072B2",
    "Annealed": "#009E73",
    "Annealed + Cold drawn": "#56B4E9",
    "Spheroidized annealed": "#CC79A7",
    "Normalized": "#D55E00",
    "Normalized + Cold drawn": "#8C8C8C",
    "Quenched & tempered": "#332288",
    "Other": "#B0B0B0",
}
MICRO_CORTA = {  # etiqueta microestructural corta para los ejes
    "Hot rolled": "ferrita + perlita",
    "Cold drawn": "↑ dislocaciones (deformado)",
    "Annealed": "perlita gruesa",
    "Annealed + Cold drawn": "perlita gruesa + deformación",
    "Spheroidized annealed": "cementita esferoidal",
    "Normalized": "perlita fina, grano refinado",
    "Normalized + Cold drawn": "perlita fina + deformación",
    "Quenched & tempered": "martensita revenida",
    "Other": "",
}
FAMILIAS = {  # familia -> (marcador, etiqueta corta, color para gráficas de composición)
    "Carbono simple 10xx": ("o", "10xx", "#4C72B0"),
    "Resulfurados/Refosforados 11xx-12xx": ("s", "11xx-12xx", "#DD8452"),
    "Alto Mn 13xx-15xx": ("^", "13xx-15xx", "#55A868"),
    "Aleados 3xxx-9xxx": ("D", "Aleados 3xxx-9xxx", "#C44E52"),
}
ROJO_ANOMALIA = "#E41A1C"

MIN_N_SUBGRUPO = 8          # n mínimo para reportar correlaciones por subgrupo
RANGO_C_MINIMO = 0.15       # si el rango de %C del subgrupo es menor, la correlación es poco informativa
REF_UTS_HB = (3.4, 3.6)     # regla empírica para aceros: UTS (MPa) ≈ 3.45 × HB

# ----------------------------------------------------------------------------
# 3. TEXTOS METALÚRGICOS (se usan en el resumen y como anotaciones en las gráficas)
# ----------------------------------------------------------------------------
TEXTO_CARBONO = [
    "En aceros hipoeutectoides (< ~0.77 %C) la microestructura es ferrita proeutectoide + perlita. "
    "La fracción de perlita crece con el %C (regla de la palanca) y la perlita (láminas alternadas de "
    "ferrita y cementita, Fe3C) es más dura y resistente que la ferrita.",
    "Más cementita y menor espaciamiento interlaminar significan más obstáculos al deslizamiento de "
    "dislocaciones: suben UTS, YS y dureza.",
    "Al disminuir la ferrita dúctil, baja la elongación (y la reducción de área).",
    "Cerca de 0.77 %C (eutectoide) la estructura es casi toda perlita; por encima (hipereutectoides, "
    "p. ej. SAE 1095) aparece cementita proeutectoide en límite de grano: sigue subiendo la dureza, "
    "pero fragiliza.",
    "La dispersión alrededor de la tendencia proviene de: aleantes (Mn, Cr, Mo, Ni), condición de "
    "procesamiento, tamaño de barra y de que los datos son valores nominales de tablas SAE.",
]

TEXTO_CONDICIONES = {
    "Hot rolled": (
        "Laminado en caliente con enfriamiento posterior al aire: ferrita + perlita de grano moderado. "
        "Es la referencia 'base' de los aceros al carbono: sus propiedades dependen sobre todo del %C."
    ),
    "Cold drawn": (
        "Deformación plástica en frío (estirado) sobre barra laminada: sube fuertemente la densidad de "
        "dislocaciones y su entrecruzamiento (endurecimiento por deformación, sigma ∝ raíz de rho) y "
        "se alargan granos y perlita. Efecto típico: ↑↑ YS, ↑ UTS (menos), ↓ elongación y reducción "
        "de área; aumenta la relación YS/UTS."
    ),
    "Annealed": (
        "Recocido: enfriamiento lento en horno, transformación cercana al equilibrio. Da perlita gruesa "
        "(mayor espaciamiento interlaminar), grano mayor y alivio de tensiones: menor dureza y "
        "resistencia, mayor ductilidad que normalizado o temple."
    ),
    "Annealed + Cold drawn": (
        "Recocido previo + estirado en frío: parte de una base blanda (perlita gruesa/esferoidizada) "
        "que luego se endurece por deformación (↑ dislocaciones, ↓ elongación)."
    ),
    "Spheroidized annealed": (
        "Recocido de esferoidización: la cementita pasa de láminas a glóbulos (Fe3C) en matriz ferrítica. "
        "Menor interfaz y menor restricción al deslizamiento: a igual %C es la estructura más blanda y de "
        "mejor maquinabilidad/conformabilidad en aceros de alto %C. En este dataset TODOS los registros son "
        "'Spheroidized annealed, cold drawn' (estirado posterior que endurece por deformación) y de %C alto, "
        "por lo que su dureza no es la mínima posible de esta microestructura."
    ),
    "Normalized": (
        "Normalizado: austenización y enfriamiento al aire (más rápido que en horno). Produce perlita "
        "más fina y grano austenítico refinado y homogéneo: UTS, YS y dureza mayores que el recocido "
        "a igual composición, con buena tenacidad."
    ),
    "Normalized + Cold drawn": (
        "Base normalizada (perlita fina) + endurecimiento por deformación: ↑ resistencia, ↓ ductilidad. "
        "n muy pequeño: solo orientativo."
    ),
    "Quenched & tempered": (
        "Temple (aceite/agua) forma martensita (C sobresaturado, alta densidad de dislocaciones); el "
        "revenido a 540 °C precipita carburos finos (Fe3C) en matriz de ferrita ('martensita revenida'): "
        "alta resistencia y dureza con ductilidad y tenacidad recuperadas. La templabilidad depende de "
        "los aleantes (Cr, Mo, Ni, Mn): en este dataset son aceros aleados."
    ),
}

LIMITACIONES = [
    "Los datos son valores nominales/típicos de tablas SAE, no ensayos individuales: 'n' es el número "
    "de renglones, no de probetas, y los grados vecinos (1010, 1012, 1015...) no son independientes, "
    "por lo que los valores p son optimistas.",
    "C_pct es el promedio de los límites Min-Max de la especificación (no el %C real de una colada); "
    "para 12 grados solo se dispone de C (Max).",
    "Proceso y composición están confundidos: recocido, normalizado y temple y revenido son casi "
    "exclusivamente aceros aleados, mientras Hot rolled / Cold drawn cubren sobre todo 10xx-15xx. "
    "Comparar medias entre grupos mezcla efecto de aleación y efecto de proceso; por eso se incluye "
    "el análisis por pares Hot rolled vs Cold drawn del mismo SAE Grade.",
    "No se registran diámetro de barra, velocidad de enfriamiento ni reducción en frío, que modulan "
    "las propiedades.",
    "Correlación no implica causalidad; las regresiones lineales son descriptivas.",
]


# ----------------------------------------------------------------------------
# 4. ARGUMENTOS Y CARGA DE DATOS
# ----------------------------------------------------------------------------
def leer_argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Análisis %C y condiciones de procesamiento.")
    p.add_argument("--anomalias", choices=["resaltar", "excluir", "incluir"], default="resaltar",
                   help="Tratamiento de registros con flag_anomalia (defecto: resaltar).")
    p.add_argument("--valores", choices=["corregidos", "originales"], default="corregidos",
                   help="UTS/YS conciliados (UTS_MPa_corr, YS_MPa_corr) u originales (defecto: corregidos).")
    p.add_argument("--min-n", type=int, default=MIN_N_SUBGRUPO,
                   help=f"n mínimo por subgrupo para reportar correlaciones (defecto: {MIN_N_SUBGRUPO}).")
    p.add_argument("--dpi", type=int, default=300, help="Resolución de las figuras (defecto: 300).")
    return p.parse_args()


def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Lee aceros_limpio.csv (obligatorio) y la auditoría (opcional)."""
    if not ARCHIVO_LIMPIO.exists():
        raise FileNotFoundError(
            f"No existe {ARCHIVO_LIMPIO.relative_to(RAIZ)}. Ejecuta primero: python src/01_limpieza_datos.py"
        )
    df = pd.read_csv(ARCHIVO_LIMPIO, encoding="utf-8-sig")

    requeridas = ["registro", "SAE Grade", "Familia_SAE", "Condicion_Grupo", "C_pct", "UTS (MPa)", "YS (MPa)",
                  "UTS_MPa_corr", "YS_MPa_corr", "Hardness_HB_num", "Elongation (%)",
                  "flag_anomalia", "motivo_auditoria"]
    faltan = [c for c in requeridas if c not in df.columns]
    if faltan:
        raise KeyError(f"Faltan columnas en aceros_limpio.csv: {faltan}. ¿Se ejecutó la versión actual del script 01?")

    df["flag_anomalia"] = df["flag_anomalia"].astype(str).str.lower().eq("true")
    df["motivo_auditoria"] = df["motivo_auditoria"].fillna("")

    auditoria = None
    if ARCHIVO_AUDITORIA.exists():
        auditoria = pd.read_csv(ARCHIVO_AUDITORIA, encoding="utf-8-sig")
    return df, auditoria


def preparar_variables(df: pd.DataFrame, usar_corregidos: bool) -> pd.DataFrame:
    """Crea las columnas de análisis UTS, YS, HB y ELONG (nombres cortos y homogéneos)."""
    df = df.copy()
    df["UTS"] = df["UTS_MPa_corr"] if usar_corregidos else df["UTS (MPa)"]
    df["YS"] = df["YS_MPa_corr"] if usar_corregidos else df["YS (MPa)"]
    df["HB"] = df["Hardness_HB_num"]          # solo escala Brinell válida (HRB excluido en el script 01)
    df["ELONG"] = df["Elongation (%)"]
    return df


def grupos_presentes(df: pd.DataFrame) -> list[str]:
    """Condiciones presentes, en orden metalúrgico lógico."""
    existentes = set(df["Condicion_Grupo"].dropna())
    return [g for g in ORDEN_CONDICIONES if g in existentes] + sorted(existentes - set(ORDEN_CONDICIONES))


# ----------------------------------------------------------------------------
# 5. ESTADÍSTICA
# ----------------------------------------------------------------------------
def p_texto(p: float) -> str:
    if pd.isna(p):
        return "NA"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fuerza_texto(r: float) -> str:
    a = abs(r)
    return "fuerte" if a >= 0.7 else "moderada" if a >= 0.4 else "débil"


def estadisticos_relacion(x: pd.Series, y: pd.Series) -> dict:
    """Pearson, Spearman y regresión lineal simple y = a + b x (con IC 95 % de la pendiente)."""
    vacio = {k: np.nan for k in (
        "pearson_r", "pearson_p", "spearman_rho", "spearman_p", "pendiente", "pendiente_ic95_inf",
        "pendiente_ic95_sup", "pendiente_por_0_1C", "intercepto", "r2")}
    m = x.notna() & y.notna()
    x, y = x[m].astype(float), y[m].astype(float)
    n = len(x)
    if n < 3 or x.nunique() < 2 or y.nunique() < 2:
        return {"n": n, **vacio}
    r, pr = stats.pearsonr(x, y)
    rho, ps = stats.spearmanr(x, y)
    lr = stats.linregress(x, y)
    t = stats.t.ppf(0.975, n - 2)
    return {
        "n": n, "pearson_r": r, "pearson_p": pr, "spearman_rho": rho, "spearman_p": ps,
        "pendiente": lr.slope, "pendiente_ic95_inf": lr.slope - t * lr.stderr,
        "pendiente_ic95_sup": lr.slope + t * lr.stderr, "pendiente_por_0_1C": lr.slope * 0.1,
        "intercepto": lr.intercept, "r2": lr.rvalue ** 2,
    }


def definir_subgrupos(df: pd.DataFrame, min_n: int) -> dict[str, pd.Series]:
    """Subgrupos para las correlaciones. Los 10xx por condición son el 'control' más limpio."""
    es_10 = df["Familia_SAE"].eq("Carbono simple 10xx")
    sub = {
        "Todos los registros": pd.Series(True, index=df.index),
        "10xx (carbono simple)": es_10,
        "10xx · Hot rolled": es_10 & df["Condicion_Grupo"].eq("Hot rolled"),
        "10xx · Cold drawn": es_10 & df["Condicion_Grupo"].eq("Cold drawn"),
        "Aleados 3xxx-9xxx": df["Familia_SAE"].eq("Aleados 3xxx-9xxx"),
    }
    for g in grupos_presentes(df):
        if (df["Condicion_Grupo"] == g).sum() >= min_n:
            sub[f"Condición: {g}"] = df["Condicion_Grupo"].eq(g)
    return sub


def tabla_correlaciones(df: pd.DataFrame, subgrupos: dict[str, pd.Series], min_n: int) -> pd.DataFrame:
    """C_pct vs cada propiedad, para cada subgrupo."""
    filas = []
    for nombre, mascara in subgrupos.items():
        d = df[mascara]
        rango_c = d["C_pct"].max() - d["C_pct"].min() if d["C_pct"].notna().any() else np.nan
        for clave, etiqueta in VARIABLES.items():
            e = estadisticos_relacion(d["C_pct"], d[clave])
            notas = []
            if e["n"] < min_n:
                notas.append(f"n < {min_n}")
            if pd.notna(rango_c) and rango_c < RANGO_C_MINIMO:
                notas.append(f"rango de %C estrecho (< {RANGO_C_MINIMO} %): correlación poco informativa")
            filas.append({"subgrupo": nombre, "clave": clave, "variable": etiqueta,
                          "rango_C": round(rango_c, 3), **e, "nota": "; ".join(notas)})
    cols = ["subgrupo", "clave", "variable", "n", "rango_C", "pearson_r", "pearson_p", "spearman_rho",
            "spearman_p", "pendiente", "pendiente_ic95_inf", "pendiente_ic95_sup",
            "pendiente_por_0_1C", "intercepto", "r2", "nota"]
    return pd.DataFrame(filas)[cols]


def tabla_sensibilidad(df_completo: pd.DataFrame) -> pd.DataFrame:
    """Compara C_pct vs propiedades con todos los registros y sin los marcados (flag_anomalia)."""
    filas = []
    conjuntos = {"todos": df_completo, "sin_anomalias": df_completo[~df_completo["flag_anomalia"]]}
    for clave, etiqueta in VARIABLES.items():
        fila = {"variable": etiqueta}
        for nombre, d in conjuntos.items():
            e = estadisticos_relacion(d["C_pct"], d[clave])
            fila.update({f"n_{nombre}": e["n"], f"pearson_r_{nombre}": e["pearson_r"],
                         f"spearman_rho_{nombre}": e["spearman_rho"],
                         f"pendiente_0_1C_{nombre}": e["pendiente_por_0_1C"]})
        fila["delta_pearson_r"] = fila["pearson_r_sin_anomalias"] - fila["pearson_r_todos"]
        fila["delta_spearman_rho"] = fila["spearman_rho_sin_anomalias"] - fila["spearman_rho_todos"]
        filas.append(fila)
    return pd.DataFrame(filas)


def estadisticas_por_condicion(df: pd.DataFrame) -> pd.DataFrame:
    """n, media, mediana, desv. est., mín y máx de cada propiedad por Condicion_Grupo."""
    variables = {**VARIABLES, "C_pct": "%C (C_pct)"}
    filas = []
    for g in grupos_presentes(df):
        d = df[df["Condicion_Grupo"] == g]
        for clave, etiqueta in variables.items():
            s = d[clave].dropna()
            filas.append({
                "Condicion_Grupo": g, "n_registros_grupo": len(d), "variable": etiqueta, "n": len(s),
                "media": s.mean() if len(s) else np.nan,
                "mediana": s.median() if len(s) else np.nan,
                "desv_est": s.std(ddof=1) if len(s) > 1 else np.nan,
                "min": s.min() if len(s) else np.nan,
                "max": s.max() if len(s) else np.nan,
            })
    return pd.DataFrame(filas)


def pares_hot_rolled_vs_cold_drawn(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Diferencias Cold drawn - Hot rolled para el MISMO SAE Grade (composición nominal idéntica).

    Este diseño pareado aísla el efecto del endurecimiento por deformación en frío de la
    composición, que en la comparación de medias entre grupos está confundida.
    """
    hr = df[df["Condicion_Grupo"] == "Hot rolled"].drop_duplicates("SAE Grade").set_index("SAE Grade")
    cd = df[df["Condicion_Grupo"] == "Cold drawn"].drop_duplicates("SAE Grade").set_index("SAE Grade")
    grados = sorted(set(hr.index) & set(cd.index))

    filas = []
    for g in grados:
        fila = {"SAE Grade": g, "Familia_SAE": hr.at[g, "Familia_SAE"], "C_pct": hr.at[g, "C_pct"],
                "flag_anomalia": bool(hr.at[g, "flag_anomalia"] or cd.at[g, "flag_anomalia"])}
        for clave in VARIABLES:
            fila[f"{clave}_HotRolled"] = hr.at[g, clave]
            fila[f"{clave}_ColdDrawn"] = cd.at[g, clave]
            fila[f"delta_{clave}"] = cd.at[g, clave] - hr.at[g, clave]
        filas.append(fila)
    detalle = pd.DataFrame(filas)

    resumen = []
    for clave, etiqueta in VARIABLES.items():
        if detalle.empty:
            break
        a, b = detalle[f"{clave}_HotRolled"], detalle[f"{clave}_ColdDrawn"]
        ok = a.notna() & b.notna()
        delta = (b - a)[ok]
        rel = ((b - a) / a * 100)[ok]
        if len(delta) >= 3:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    p_w = stats.wilcoxon(delta).pvalue
                except ValueError:
                    p_w = np.nan
                p_t = stats.ttest_rel(b[ok], a[ok]).pvalue
        else:
            p_w = p_t = np.nan
        resumen.append({
            "variable": etiqueta, "n_pares": int(ok.sum()), "delta_medio": delta.mean(),
            "delta_mediana": delta.median(), "delta_desv_est": delta.std(ddof=1),
            "delta_min": delta.min(), "delta_max": delta.max(),
            "delta_relativo_medio_pct": rel.mean(),
            "pares_con_aumento": int((delta > 0).sum()), "pares_con_descenso": int((delta < 0).sum()),
            "p_wilcoxon": p_w, "p_ttest_pareado": p_t,
        })
    return detalle, pd.DataFrame(resumen)


# ----------------------------------------------------------------------------
# 6. UTILIDADES GRÁFICAS
# ----------------------------------------------------------------------------
def configurar_estilo() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "axes.titlesize": 11, "axes.labelsize": 11, "legend.fontsize": 9,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def mpl(texto: str) -> str:
    """Escribe Fe3C como Fe$_3$C (subíndice) para matplotlib."""
    return texto.replace("Fe3C", "Fe$_3$C")


def pie_de_figura(fig: plt.Figure, texto: str, ancho: int = 165, y: float = 0.012) -> None:
    fig.text(0.5, y, mpl(textwrap.fill(texto, ancho)), ha="center", va="bottom", fontsize=8.6,
             color="#333333", style="italic")


def guardar_figura(fig: plt.Figure, nombre: str, dpi: int) -> Path:
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    ruta = DIR_FIGURAS / nombre
    fig.savefig(ruta, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return ruta


def boxplot_horizontal(ax: plt.Axes, datos, posicion: int, **kwargs):
    """Caja horizontal compatible con matplotlib < 3.10 (vert=False) y >= 3.10 (orientation)."""
    try:
        return ax.boxplot(datos, positions=[posicion], orientation="horizontal", **kwargs)
    except TypeError:
        return ax.boxplot(datos, positions=[posicion], vert=False, **kwargs)


def dispersion_c(ax: plt.Axes, d: pd.DataFrame, clave: str, resaltar: bool, tam: float = 46) -> None:
    """Dispersión %C vs propiedad: color = condición, marcador = familia SAE, aro rojo = anomalía."""
    validos = d["C_pct"].notna() & d[clave].notna()
    for g in grupos_presentes(d):
        for fam, (marcador, _, _) in FAMILIAS.items():
            s = d[validos & (d["Condicion_Grupo"] == g) & (d["Familia_SAE"] == fam)]
            if not s.empty:
                ax.scatter(s["C_pct"], s[clave], c=COLORES_CONDICION.get(g, "#B0B0B0"), marker=marcador,
                           s=tam, alpha=0.85, edgecolors="white", linewidths=0.6, zorder=3)
    if resaltar:
        f = d[validos & d["flag_anomalia"]]
        if not f.empty:
            ax.scatter(f["C_pct"], f[clave], s=210, facecolors="none", edgecolors=ROJO_ANOMALIA,
                       linewidths=1.8, zorder=4)


def recta_ajuste(ax: plt.Axes, x: pd.Series, y: pd.Series, color: str = "black",
                 estilo: str = "--", ancho: float = 1.8) -> dict:
    """Dibuja la recta de regresión y devuelve los estadísticos."""
    e = estadisticos_relacion(x, y)
    if e["n"] >= 3 and pd.notna(e["pendiente"]):
        m = x.notna() & y.notna()
        xs = np.array([x[m].min(), x[m].max()])
        ax.plot(xs, e["intercepto"] + e["pendiente"] * xs, color=color, ls=estilo, lw=ancho, zorder=5)
    return e


def anotar_estadisticos(ax: plt.Axes, e: dict, clave: str) -> None:
    """Cuadro con r, rho, n y pendiente por cada +0.1 %C (posición según el signo esperado)."""
    if e["n"] < 3 or pd.isna(e["pearson_r"]):
        return
    texto = (f"Pearson r = {e['pearson_r']:+.2f} (p {p_texto(e['pearson_p'])})\n"
             f"Spearman ρ = {e['spearman_rho']:+.2f}\n"
             f"R² = {e['r2']:.2f} | n = {e['n']}\n"
             f"Pendiente: {e['pendiente_por_0_1C']:+.1f} {UNIDAD_CORTA[clave]} por +0.1 %C")
    derecha = e["pendiente"] < 0
    ax.text(0.97 if derecha else 0.03, 0.97, texto, transform=ax.transAxes, va="top",
            ha="right" if derecha else "left", fontsize=8.8, zorder=6,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#999999", alpha=0.93))


def leyenda_condiciones(d: pd.DataFrame, resaltar: bool, con_familias: bool = True,
                        con_ajuste: bool = True) -> list:
    """Manijas de leyenda: color = condición, marcador = familia, aro rojo = anomalía, recta = ajuste."""
    h = [Line2D([], [], marker="o", ls="", color=COLORES_CONDICION.get(g, "#B0B0B0"), label=g,
                markeredgecolor="white", markersize=9) for g in grupos_presentes(d)]
    if con_familias:
        usadas = set(d["Familia_SAE"].dropna())
        h += [Line2D([], [], marker=mk, ls="", color="#555555", label=f"Familia {lab}", markersize=8)
              for fam, (mk, lab, _) in FAMILIAS.items() if fam in usadas]
    if resaltar and d["flag_anomalia"].any():
        h.append(Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=ROJO_ANOMALIA,
                        markeredgewidth=1.8, markersize=13, label="Registro con flag_anomalia"))
    if con_ajuste:
        h.append(Line2D([], [], color="black", ls="--", lw=1.8, label="Regresión lineal"))
    return h


def nota_modo(modo: str, n_excluidos: int) -> str:
    if modo == "excluir":
        return f"Se excluyeron {n_excluidos} registros con flag_anomalia. "
    if modo == "resaltar":
        return "Aro rojo = registro con flag_anomalia (valores de resistencia conciliados con ksi cuando procede). "
    return ""


# ----------------------------------------------------------------------------
# 7. FIGURAS
# ----------------------------------------------------------------------------
def fig_c_vs_propiedades(d: pd.DataFrame, resaltar: bool, modo: str, n_excl: int, dpi: int) -> Path:
    """Figura 1: %C vs UTS, YS, HB y elongación (todos los registros)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, (clave, etiqueta) in zip(axes.ravel(), VARIABLES.items()):
        dispersion_c(ax, d, clave, resaltar)
        e = recta_ajuste(ax, d["C_pct"], d[clave])
        anotar_estadisticos(ax, e, clave)
        ax.set_title(mpl(TITULO_PANEL[clave]), loc="left", fontsize=10.5)
        ax.set_xlabel("Carbono, %C (promedio de límites Min-Max, % en masa)")
        ax.set_ylabel(etiqueta)
    fig.suptitle("Efecto del carbono en las propiedades mecánicas de aceros al carbono y aleados",
                 fontsize=14, fontweight="bold", y=0.985)
    fig.legend(handles=leyenda_condiciones(d, resaltar), loc="lower center", ncol=4, frameon=True,
               bbox_to_anchor=(0.5, 0.075))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.22, hspace=0.42, wspace=0.2)
    pie_de_figura(fig, "Metalurgia: al subir el %C aumenta la fracción de perlita (ferrita + cementita Fe3C; ~100 % "
                       "perlita en el eutectoide, ~0.77 %C). La cementita y el menor espaciamiento interlaminar "
                       "frenan a las dislocaciones: suben UTS, YS y dureza, y la menor ferrita dúctil reduce la "
                       "elongación. La dispersión vertical refleja aleantes, proceso (color) y que son valores "
                       "nominales de tablas SAE. " + nota_modo(modo, n_excl))
    return guardar_figura(fig, "fig02_01_carbono_vs_propiedades.png", dpi)


def fig_c_vs_propiedades_10xx(d: pd.DataFrame, resaltar: bool, modo: str, n_excl: int, dpi: int) -> Path:
    """Figura 2: solo carbono simple 10xx, Hot rolled vs Cold drawn (control de composición)."""
    base = d[d["Familia_SAE"].eq("Carbono simple 10xx")]
    series = [("Hot rolled", COLORES_CONDICION["Hot rolled"]), ("Cold drawn", COLORES_CONDICION["Cold drawn"])]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, (clave, etiqueta) in zip(axes.ravel(), VARIABLES.items()):
        for i, (g, color) in enumerate(series):
            s = base[base["Condicion_Grupo"] == g]
            v = s["C_pct"].notna() & s[clave].notna()
            ax.scatter(s.loc[v, "C_pct"], s.loc[v, clave], c=color, s=50, alpha=0.85,
                       edgecolors="white", linewidths=0.6, zorder=3)
            if resaltar:
                f = s[v & s["flag_anomalia"]]
                if not f.empty:
                    ax.scatter(f["C_pct"], f[clave], s=210, facecolors="none", edgecolors=ROJO_ANOMALIA,
                               linewidths=1.8, zorder=4)
            e = recta_ajuste(ax, s["C_pct"], s[clave], color=color, estilo="-", ancho=2.2)
            if e["n"] >= 3 and pd.notna(e["pearson_r"]):
                ax.text(0.03, 0.97 - 0.085 * i,
                        f"{g}: r = {e['pearson_r']:+.2f}, {e['pendiente_por_0_1C']:+.1f} {UNIDAD_CORTA[clave]}/0.1 %C "
                        f"(n = {e['n']})", transform=ax.transAxes, va="top", ha="left", fontsize=8.8,
                        color=color, fontweight="bold", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))
        ax.set_title(f"({'abcd'[list(VARIABLES).index(clave)]}) {etiqueta}", loc="left", fontsize=11)
        ax.set_xlabel("Carbono, %C (promedio de límites Min-Max, % en masa)")
        ax.set_ylabel(etiqueta)
    fig.suptitle("Carbono simple 10xx: %C vs propiedades, Hot rolled frente a Cold drawn",
                 fontsize=14, fontweight="bold", y=0.985)
    h = [Line2D([], [], marker="o", ls="-", color=c, label=g, markersize=8, lw=2.2) for g, c in series]
    if resaltar and base["flag_anomalia"].any():
        h.append(Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=ROJO_ANOMALIA,
                        markeredgewidth=1.8, markersize=13, label="Registro con flag_anomalia"))
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 0.075))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.2, hspace=0.32, wspace=0.2)
    pie_de_figura(fig, "A igual %C, la separación vertical entre rectas es el efecto del proceso: el estirado en frío "
                       "aumenta la densidad de dislocaciones (endurecimiento por deformación), elevando YS y UTS y "
                       "reduciendo la elongación. La pendiente de cada recta refleja el efecto del %C (más perlita/Fe3C). "
                       "Se controla la familia de aleación al usar solo 10xx. " + nota_modo(modo, n_excl))
    return guardar_figura(fig, "fig02_02_carbono_10xx_hot_rolled_vs_cold_drawn.png", dpi)


def fig_boxplots_condicion(d: pd.DataFrame, resaltar: bool, modo: str, n_excl: int, dpi: int) -> Path:
    """Figura 3: distribución de cada propiedad por condición (caja + puntos + media)."""
    grupos = grupos_presentes(d)
    rng = np.random.default_rng(42)  # semilla fija: jitter reproducible
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    for ax, (clave, etiqueta) in zip(axes.ravel(), VARIABLES.items()):
        etiquetas = []
        for pos, g in enumerate(grupos):
            s = d[d["Condicion_Grupo"] == g].dropna(subset=[clave])
            etiquetas.append(f"{g}\n{MICRO_CORTA.get(g, '')}\n(n = {len(s)})")
            if s.empty:
                continue
            color = COLORES_CONDICION.get(g, "#B0B0B0")
            bp = boxplot_horizontal(ax, s[clave].values, pos, widths=0.55, patch_artist=True,
                                    showfliers=False, manage_ticks=False,
                                    medianprops=dict(color="black", lw=2), whiskerprops=dict(color="#555555"),
                                    capprops=dict(color="#555555"), boxprops=dict(edgecolor="#555555"))
            bp["boxes"][0].set_facecolor(color)
            bp["boxes"][0].set_alpha(0.35)
            ax.scatter(s[clave], pos + rng.uniform(-0.2, 0.2, len(s)), c=color, s=20, alpha=0.85,
                       edgecolors="white", linewidths=0.4, zorder=3)
            ax.scatter([s[clave].mean()], [pos], marker="D", s=70, c="white", edgecolors="black",
                       linewidths=1.3, zorder=5)
            if resaltar:
                f = s[s["flag_anomalia"]]
                if not f.empty:
                    ax.scatter(f[clave], [pos] * len(f), s=140, facecolors="none", edgecolors=ROJO_ANOMALIA,
                               linewidths=1.8, zorder=6)
        ax.set_yticks(range(len(grupos)))
        ax.set_yticklabels(etiquetas, fontsize=8.6)
        ax.invert_yaxis()
        ax.set_xlabel(etiqueta)
        ax.set_title(f"({'abcd'[list(VARIABLES).index(clave)]}) {etiqueta} por condición de procesamiento",
                     loc="left", fontsize=11)
    h = [Line2D([], [], marker="D", ls="", markerfacecolor="white", markeredgecolor="black",
                label="Media", markersize=8),
         Line2D([], [], color="black", lw=2, label="Mediana"),
         Patch(facecolor="#BBBBBB", edgecolor="#555555", alpha=0.5, label="Rango intercuartílico (caja)")]
    if resaltar and d["flag_anomalia"].any():
        h.append(Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=ROJO_ANOMALIA,
                        markeredgewidth=1.8, markersize=11, label="Registro con flag_anomalia"))
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 0.075))
    fig.suptitle("Propiedades mecánicas según la condición de procesamiento y su microestructura",
                 fontsize=14, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.92, bottom=0.19, hspace=0.3, wspace=0.42)
    pie_de_figura(fig, "Lectura metalúrgica: el estirado en frío endurece por deformación (↑ YS/UTS, ↓ elongación); "
                       "normalizado (perlita fina) supera al recocido (perlita gruesa); la esferoidización deja la "
                       "cementita en glóbulos (estructura blanda); temple y revenido da martensita revenida "
                       "(alta resistencia con ductilidad recuperada). Las condiciones difieren también en %C y "
                       "aleación (ver fig02_05), así que las diferencias de medias no son solo efecto del proceso. "
                       + nota_modo(modo, n_excl), ancho=175)
    return guardar_figura(fig, "fig02_03_propiedades_por_condicion.png", dpi)


def fig_pares_hr_cd(detalle: pd.DataFrame, resumen: pd.DataFrame, resaltar: bool, dpi: int) -> Path | None:
    """Figura 4: pares Hot rolled -> Cold drawn del mismo SAE Grade."""
    if detalle.empty:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(13, 11.5))
    for ax, (clave, etiqueta) in zip(axes.ravel(), VARIABLES.items()):
        a, b = detalle[f"{clave}_HotRolled"], detalle[f"{clave}_ColdDrawn"]
        ok = a.notna() & b.notna()
        for i in detalle.index[ok]:
            sube = b[i] >= a[i]
            ax.plot([0, 1], [a[i], b[i]], color="#0072B2" if sube else "#D55E00", alpha=0.45, lw=1.2, zorder=2)
        ax.scatter(np.zeros(ok.sum()), a[ok], c=COLORES_CONDICION["Hot rolled"], s=28, zorder=3,
                   edgecolors="white", linewidths=0.4)
        ax.scatter(np.ones(ok.sum()), b[ok], c=COLORES_CONDICION["Cold drawn"], s=28, zorder=3,
                   edgecolors="white", linewidths=0.4)
        if resaltar:
            f = detalle[ok & detalle["flag_anomalia"]]
            for i in f.index:
                ax.scatter([0, 1], [a[i], b[i]], s=170, facecolors="none", edgecolors=ROJO_ANOMALIA,
                           linewidths=1.8, zorder=6)
        if ok.any():
            ax.plot([0, 1], [a[ok].mean(), b[ok].mean()], color="black", lw=3.2, marker="D", ms=8,
                    mfc="white", zorder=7)
        fila = resumen[resumen["variable"] == etiqueta]
        if not fila.empty:
            r = fila.iloc[0]
            ax.text(0.5, 0.97, f"Δ medio = {r['delta_medio']:+.1f} {UNIDAD_CORTA[clave]} "
                               f"({r['delta_relativo_medio_pct']:+.0f} %)\n"
                               f"{int(r['pares_con_aumento'])}/{int(r['n_pares'])} pares suben | "
                               f"Wilcoxon p {p_texto(r['p_wilcoxon'])}",
                    transform=ax.transAxes, ha="center", va="top", fontsize=9, zorder=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999999", alpha=0.93))
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Hot rolled", "Cold drawn"], fontsize=10.5)
        ax.set_xlim(-0.35, 1.35)
        ax.set_ylabel(etiqueta)
        ax.set_title(f"({'abcd'[list(VARIABLES).index(clave)]}) {etiqueta}", loc="left", fontsize=11)
    fig.suptitle("Efecto del estirado en frío a composición idéntica (mismo SAE Grade)",
                 fontsize=14, fontweight="bold", y=0.985)
    h = [Line2D([], [], color="#0072B2", lw=2, label="Par con aumento"),
         Line2D([], [], color="#D55E00", lw=2, label="Par con descenso"),
         Line2D([], [], color="black", lw=3, marker="D", mfc="white", label="Media de pares")]
    if resaltar and detalle["flag_anomalia"].any():
        h.append(Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=ROJO_ANOMALIA,
                        markeredgewidth=1.8, markersize=12, label="Par con flag_anomalia"))
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 0.065))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.17, hspace=0.3, wspace=0.25)
    pie_de_figura(fig, "Cada línea une el mismo grado SAE en Hot rolled y Cold drawn, por lo que la composición "
                       "nominal es idéntica y la diferencia se atribuye al proceso: la deformación plástica en frío "
                       "multiplica y enreda dislocaciones (endurecimiento por deformación), elevando sobre todo YS, "
                       "y consume ductilidad (↓ elongación).", ancho=150)
    return guardar_figura(fig, "fig02_04_pares_hot_rolled_vs_cold_drawn.png", dpi)


def fig_confusion_composicion(d: pd.DataFrame, dpi: int) -> Path:
    """Figura 5: %C y familia SAE por condición (muestra la confusión proceso-composición)."""
    grupos = grupos_presentes(d)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1, 1.1]})

    etiquetas = []
    for pos, g in enumerate(grupos):
        s = d[d["Condicion_Grupo"] == g]["C_pct"].dropna()
        etiquetas.append(f"{g}\n(n = {len(s)})")
        if s.empty:
            continue
        color = COLORES_CONDICION.get(g, "#B0B0B0")
        bp = boxplot_horizontal(ax1, s.values, pos, widths=0.55, patch_artist=True, showfliers=False,
                                manage_ticks=False, medianprops=dict(color="black", lw=2))
        bp["boxes"][0].set_facecolor(color)
        bp["boxes"][0].set_alpha(0.4)
        ax1.scatter(s, [pos] * len(s), c=color, s=22, alpha=0.7, edgecolors="white", linewidths=0.4, zorder=3)
        ax1.scatter([s.mean()], [pos], marker="D", s=65, c="white", edgecolors="black", linewidths=1.2, zorder=5)
    ax1.set_yticks(range(len(grupos)))
    ax1.set_yticklabels(etiquetas, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel("Carbono, %C (promedio de límites Min-Max, % en masa)")
    ax1.set_title("(a) %C por condición de procesamiento (rombo = media)", loc="left")

    tabla = pd.crosstab(d["Condicion_Grupo"], d["Familia_SAE"]).reindex(grupos).fillna(0)
    izquierda = np.zeros(len(grupos))
    for fam, (_, lab, color) in FAMILIAS.items():
        if fam in tabla.columns:
            ax2.barh(range(len(grupos)), tabla[fam].values, left=izquierda, color=color, label=lab,
                     edgecolor="white")
            izquierda += tabla[fam].values
    ax2.set_yticks(range(len(grupos)))
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xlabel("Número de registros")
    ax2.set_title("(b) Familia SAE dentro de cada condición", loc="left")
    ax2.legend(title="Familia SAE", loc="lower right")

    pct_al = (tabla.get("Aleados 3xxx-9xxx", pd.Series(0, index=tabla.index)) / tabla.sum(axis=1) * 100).round(0)
    detalle = ", ".join(f"{g}: {int(p)} %" for g, p in pct_al.items() if g in
                        ("Annealed", "Normalized", "Quenched & tempered", "Hot rolled", "Cold drawn"))
    fig.suptitle("Confusión entre proceso y composición: por qué no basta comparar medias entre condiciones",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.88, bottom=0.2, wspace=0.08)
    pie_de_figura(fig, f"Porcentaje de aceros aleados (3xxx-9xxx) por condición: {detalle}. "
                       "Recocido, normalizado y temple y revenido son casi solo aleados; Hot rolled y Cold drawn "
                       "son sobre todo 10xx-15xx. Cualquier diferencia de medias mezcla efecto de aleación (templabilidad, "
                       "endurecimiento por solución sólida) y de proceso (microestructura).", ancho=160)
    return guardar_figura(fig, "fig02_05_confusion_proceso_composicion.png", dpi)


def fig_heatmap_spearman(corr: pd.DataFrame, min_n: int, dpi: int) -> Path:
    """Figura 6: Spearman %C-propiedad por subgrupo (con n; † = poco informativo)."""
    subgrupos = list(dict.fromkeys(corr["subgrupo"]))
    rho = corr.pivot(index="subgrupo", columns="clave", values="spearman_rho").reindex(subgrupos)[list(VARIABLES)]
    nn = corr.pivot(index="subgrupo", columns="clave", values="n").reindex(subgrupos)[list(VARIABLES)]
    nota = corr.pivot(index="subgrupo", columns="clave", values="nota").reindex(subgrupos)[list(VARIABLES)]

    anot = np.empty(rho.shape, dtype=object)
    for i in range(rho.shape[0]):
        for j in range(rho.shape[1]):
            v = rho.iat[i, j]
            marca = "†" if isinstance(nota.iat[i, j], str) and nota.iat[i, j] else ""
            anot[i, j] = "NA" if pd.isna(v) else f"{v:+.2f}{marca}\n(n={int(nn.iat[i, j])})"

    fig, ax = plt.subplots(figsize=(10.5, 0.75 * len(subgrupos) + 3.2))
    sns.heatmap(rho.astype(float), annot=anot, fmt="", cmap="RdBu_r", vmin=-1, vmax=1, center=0,
                linewidths=0.8, linecolor="white", cbar_kws={"label": "Spearman ρ (%C vs propiedad)"}, ax=ax)
    # Celdas sin correlación calculable (p. ej. propiedad constante en el subgrupo): se rotulan explícitamente
    for i in range(rho.shape[0]):
        for j in range(rho.shape[1]):
            if pd.isna(rho.iat[i, j]):
                ax.text(j + 0.5, i + 0.5, f"n/a\n(valor constante,\nn={int(nn.iat[i, j])})", ha="center",
                        va="center", fontsize=8.5, color="#555555")
    ax.set_xticklabels([VARIABLES[c] for c in rho.columns], rotation=0)
    ax.set_yticklabels(rho.index, rotation=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Correlación de Spearman entre %C y propiedades, por subgrupo", fontsize=13, fontweight="bold", pad=14)
    fig.subplots_adjust(left=0.25, right=0.95, top=0.93, bottom=0.1)
    pie_de_figura(fig, f"† = n < {min_n} o rango de %C < {RANGO_C_MINIMO} % dentro del subgrupo (correlación poco informativa). "
                       "Rojo = relación positiva, azul = negativa. Se espera ρ > 0 para UTS, YS y HB y ρ < 0 para elongación "
                       "(más perlita/Fe3C, menos ferrita dúctil). Los subgrupos con %C casi constante (p. ej. un solo "
                       "grado de aleación) no pueden mostrar el efecto del carbono. n/a = propiedad constante en el subgrupo.", ancho=135, y=0.0)
    return guardar_figura(fig, "fig02_06_heatmap_correlaciones_por_subgrupo.png", dpi)


# ----------------------------------------------------------------------------
# 8. RESUMEN INTERPRETADO
# ----------------------------------------------------------------------------
def lista_anomalias(d: pd.DataFrame) -> list[str]:
    """Una línea por registro marcado, con sus categorías de ANOMALIA."""
    lineas = []
    for _, f in d[d["flag_anomalia"]].iterrows():
        cats = sorted(set(re.findall(r"\[ANOMALIA:([A-Z_]+)\]", f["motivo_auditoria"])))
        lineas.append(f"  #{int(f['registro']):>3}  SAE {f['SAE Grade']:<6} {f['Condicion_Grupo']:<24} "
                      f"{', '.join(cats)}")
    return lineas


def construir_resumen(args, df_completo, d, corr, est, detalle_pares, resumen_pares, sens, auditoria, figuras,
                      tablas) -> str:
    L: list[str] = []
    sep = "=" * 92

    def titulo(t: str) -> None:
        L.extend(["", sep, t, sep])

    L.append(sep)
    L.append("RESUMEN 02 · %C Y CONDICIONES DE PROCESAMIENTO")
    L.append(sep)
    L.append(f"Valores de resistencia : {args.valores} | Tratamiento de anomalías: {args.anomalias}")
    n_flag = int(df_completo["flag_anomalia"].sum())
    L.append(f"Registros totales     : {len(df_completo)} | con flag_anomalia: {n_flag} | "
             f"en el análisis: {len(d)}")
    if auditoria is not None:
        L.append(f"Auditoría (script 01)  : {len(auditoria)} registros con hallazgos (anomalías, correcciones, avisos)")
    if args.valores == "originales" and args.anomalias != "excluir":
        L.append("AVISO: con valores originales y sin excluir anomalías, los errores de captura "
                 "(p. ej. YS > UTS) distorsionan las estadísticas.")
    if n_flag:
        L.append("Registros marcados:")
        L.extend(lista_anomalias(df_completo))

    # --- 1. Carbono vs propiedades -------------------------------------------------------
    titulo("1. %C vs PROPIEDADES MECÁNICAS")
    L.append("Base metalúrgica:")
    for t in TEXTO_CARBONO:
        L.append(textwrap.fill(t, 90, initial_indent="  - ", subsequent_indent="    "))

    glob = corr[corr["subgrupo"] == "Todos los registros"]
    L.append("\nResultados (todos los registros; pendiente = cambio por cada +0.10 %C):")
    for _, r in glob.iterrows():
        esperado = SIGNO_ESPERADO[r["clave"]]
        coincide = np.sign(r["spearman_rho"]) == esperado and r["spearman_p"] < 0.05
        L.append(f"  {r['variable']:<22} n={int(r['n']):>3}  Pearson r={r['pearson_r']:+.2f} (p {p_texto(r['pearson_p'])})  "
                 f"Spearman ρ={r['spearman_rho']:+.2f}  R²={r['r2']:.2f}  "
                 f"pendiente={r['pendiente_por_0_1C']:+.1f} {UNIDAD_CORTA[r['clave']]}/0.1 %C  "
                 f"-> relación {fuerza_texto(r['spearman_rho'])}, "
                 f"{'consistente' if coincide else 'NO consistente'} con lo esperado")

    ctrl = corr[corr["subgrupo"] == "10xx · Hot rolled"]
    if not ctrl.empty and ctrl["n"].iloc[0] >= 3:
        L.append("\nControl metalúrgico (10xx · Hot rolled: misma familia y mismo proceso, solo varía el %C):")
        for _, r in ctrl.iterrows():
            L.append(f"  {r['variable']:<22} n={int(r['n']):>3}  r={r['pearson_r']:+.2f}  ρ={r['spearman_rho']:+.2f}  "
                     f"pendiente={r['pendiente_por_0_1C']:+.1f} {UNIDAD_CORTA[r['clave']]}/0.1 %C "
                     f"(IC95 % por 0.1 %C: {r['pendiente_ic95_inf'] * 0.1:+.1f} a {r['pendiente_ic95_sup'] * 0.1:+.1f})")
        L.append(textwrap.fill(
            "Lectura: si las pendientes del control son más limpias (mayor R²) que las globales, la dispersión "
            "global se debe a aleantes y proceso, no a ausencia de efecto del carbono.", 90,
            initial_indent="  ", subsequent_indent="  "))

    estrechos = corr[(corr["nota"] != "") & (corr["clave"] == "UTS")]
    if not estrechos.empty:
        L.append("\nSubgrupos con correlación poco informativa (†):")
        for _, r in estrechos.iterrows():
            L.append(f"  - {r['subgrupo']}: {r['nota']}")

    uh = d.dropna(subset=["UTS", "HB"])
    if len(uh):
        ratio = uh["UTS"] / uh["HB"]
        fuera = int(((ratio < 2.8) | (ratio > 4.2)).sum())
        L.append(f"\nVerificación de coherencia UTS/HB: mediana = {ratio.median():.2f} MPa/HB "
                 f"(rango intercuartílico {ratio.quantile(.25):.2f}-{ratio.quantile(.75):.2f}); "
                 f"referencia empírica en aceros ≈ {REF_UTS_HB[0]}-{REF_UTS_HB[1]}. "
                 f"Registros fuera de 2.8-4.2: {fuera}.")

    L.append("\nSensibilidad a anomalías (Δ al excluir registros con flag_anomalia):")
    for _, r in sens.iterrows():
        L.append(f"  {r['variable']:<22} r: {r['pearson_r_todos']:+.3f} -> {r['pearson_r_sin_anomalias']:+.3f} "
                 f"(Δ {r['delta_pearson_r']:+.3f}) | ρ: {r['spearman_rho_todos']:+.3f} -> "
                 f"{r['spearman_rho_sin_anomalias']:+.3f} (Δ {r['delta_spearman_rho']:+.3f})")

    # --- 2. Condiciones -----------------------------------------------------------------
    titulo("2. COMPARACIÓN ENTRE CONDICIONES DE PROCESAMIENTO")
    L.append("Media ± desv. est. (n) por condición:")
    piv = {}
    for g in grupos_presentes(d):
        partes = []
        for etiqueta in list(VARIABLES.values()) + ["%C (C_pct)"]:
            f = est[(est["Condicion_Grupo"] == g) & (est["variable"] == etiqueta)].iloc[0]
            corto = etiqueta.split(" (")[0].replace("Dureza Brinell", "HB").replace("Elongación", "Elong")
            if f["n"] == 0:
                partes.append(f"{corto} NA")
            elif etiqueta.startswith("%C"):
                partes.append(f"%C {f['media']:.2f}")
            else:
                sd = "NA" if pd.isna(f["desv_est"]) else f"{f['desv_est']:.0f}" if corto != "Elong" else f"{f['desv_est']:.1f}"
                valor = f"{f['media']:.0f}" if corto != "Elong" else f"{f['media']:.1f}"
                partes.append(f"{corto} {valor}±{sd}")
        piv[g] = f"  {g:<24} (n={int(est[est['Condicion_Grupo'] == g]['n_registros_grupo'].iloc[0]):>2}) " + " | ".join(partes)
    L.extend(piv.values())

    L.append("\nInterpretación metalúrgica por condición:")
    for g in grupos_presentes(d):
        if g in TEXTO_CONDICIONES:
            L.append(textwrap.fill(f"{g}: {TEXTO_CONDICIONES[g]}", 90, initial_indent="  - ", subsequent_indent="    "))

    L.append("\nEfecto del estirado en frío a composición idéntica (pares del mismo SAE Grade, Hot rolled -> Cold drawn):")
    if resumen_pares.empty:
        L.append("  No hay pares disponibles.")
    else:
        for _, r in resumen_pares.iterrows():
            L.append(f"  {r['variable']:<22} pares={int(r['n_pares']):>2}  Δ medio={r['delta_medio']:+7.1f} "
                     f"({r['delta_relativo_medio_pct']:+.0f} %)  mediana={r['delta_mediana']:+7.1f}  "
                     f"suben {int(r['pares_con_aumento'])}/{int(r['n_pares'])}  "
                     f"Wilcoxon p {p_texto(r['p_wilcoxon'])}  t pareado p {p_texto(r['p_ttest_pareado'])}")
        ys_f = resumen_pares[resumen_pares["variable"] == "YS (MPa)"]
        uts_f = resumen_pares[resumen_pares["variable"] == "UTS (MPa)"]
        if not ys_f.empty and not uts_f.empty:
            mayor = ys_f.iloc[0]["delta_medio"] > uts_f.iloc[0]["delta_medio"]
            L.append(textwrap.fill(
                f"Lectura: el aumento medio de YS ({ys_f.iloc[0]['delta_medio']:+.0f} MPa) "
                f"{'supera' if mayor else 'no supera'} al de UTS ({uts_f.iloc[0]['delta_medio']:+.0f} MPa), "
                f"{'como se espera del endurecimiento por deformación (sube la relación YS/UTS)' if mayor else 'lo que difiere de lo esperado: revisar datos'}.",
                90, initial_indent="  ", subsequent_indent="  "))

    L.append("\nPrecaución al comparar los demás grupos:")
    ref = est[(est["Condicion_Grupo"] == "Hot rolled") & (est["variable"] == "%C (C_pct)")]
    c_ref = ref["media"].iloc[0] if not ref.empty else np.nan
    for g in grupos_presentes(d):
        if g == "Hot rolled":
            continue
        f = est[(est["Condicion_Grupo"] == g) & (est["variable"] == "%C (C_pct)")].iloc[0]
        if pd.notna(c_ref) and pd.notna(f["media"]) and abs(f["media"] - c_ref) > 0.10:
            L.append(f"  - {g}: %C medio {f['media']:.2f} vs {c_ref:.2f} en Hot rolled (Δ {f['media'] - c_ref:+.2f}); "
                     "parte de la diferencia en resistencia/dureza se debe al carbono, no al proceso.")
    pct_al = d.groupby("Condicion_Grupo")["Familia_SAE"].apply(lambda s: (s == "Aleados 3xxx-9xxx").mean() * 100)
    altos = [f"{g} ({p:.0f} %)" for g, p in pct_al.items() if p >= 80]
    if altos:
        L.append(textwrap.fill("  - Condiciones casi exclusivamente de aceros aleados: " + ", ".join(altos) +
                               ". Su diferencia con Hot rolled/Cold drawn mezcla aleación y proceso.", 90,
                               subsequent_indent="    "))

    titulo("3. LIMITACIONES")
    for t in LIMITACIONES:
        L.append(textwrap.fill(t, 90, initial_indent="  - ", subsequent_indent="    "))

    titulo("4. ARCHIVOS GENERADOS")
    for t in tablas:
        L.append(f"  - {t.relative_to(RAIZ)}")
    for f in figuras:
        L.append(f"  - {f.relative_to(RAIZ)}")
    return "\n".join(L)


# ----------------------------------------------------------------------------
# 9. PROGRAMA PRINCIPAL
# ----------------------------------------------------------------------------
def guardar_tabla(df: pd.DataFrame, nombre: str) -> Path:
    DIR_PROCESSED.mkdir(parents=True, exist_ok=True)
    ruta = DIR_PROCESSED / nombre
    df.to_csv(ruta, index=False, encoding="utf-8-sig")
    return ruta


def main() -> None:
    args = leer_argumentos()
    configurar_estilo()

    df_raw, auditoria = cargar_datos()
    df_completo = preparar_variables(df_raw, usar_corregidos=(args.valores == "corregidos"))

    # Conjunto del análisis principal según el modo de tratamiento de anomalías
    d = df_completo[~df_completo["flag_anomalia"]].copy() if args.anomalias == "excluir" else df_completo.copy()
    n_excl = len(df_completo) - len(d)
    resaltar = args.anomalias == "resaltar"

    # --- Tablas ----------------------------------------------------------------------
    subgrupos = definir_subgrupos(d, args.min_n)
    corr = tabla_correlaciones(d, subgrupos, args.min_n)
    est = estadisticas_por_condicion(d)
    detalle_pares, resumen_pares = pares_hot_rolled_vs_cold_drawn(d)
    sens = tabla_sensibilidad(df_completo)

    tablas = [
        guardar_tabla(corr, "tabla_02_correlaciones_c_vs_propiedades.csv"),
        guardar_tabla(est, "tabla_02_estadisticas_por_condicion.csv"),
        guardar_tabla(detalle_pares, "tabla_02_pares_hot_rolled_vs_cold_drawn.csv"),
        guardar_tabla(resumen_pares, "tabla_02_resumen_pares_hot_rolled_vs_cold_drawn.csv"),
        guardar_tabla(sens, "tabla_02_sensibilidad_anomalias.csv"),
    ]

    # --- Figuras ---------------------------------------------------------------------
    figuras = [
        fig_c_vs_propiedades(d, resaltar, args.anomalias, n_excl, args.dpi),
        fig_c_vs_propiedades_10xx(d, resaltar, args.anomalias, n_excl, args.dpi),
        fig_boxplots_condicion(d, resaltar, args.anomalias, n_excl, args.dpi),
    ]
    f4 = fig_pares_hr_cd(detalle_pares, resumen_pares, resaltar, args.dpi)
    if f4 is not None:
        figuras.append(f4)
    figuras.append(fig_confusion_composicion(d, args.dpi))
    figuras.append(fig_heatmap_spearman(corr, args.min_n, args.dpi))

    # --- Resumen ---------------------------------------------------------------------
    resumen = construir_resumen(args, df_completo, d, corr, est, detalle_pares, resumen_pares, sens,
                                auditoria, figuras, tablas)
    DIR_OUTPUTS.mkdir(parents=True, exist_ok=True)
    ruta_resumen = DIR_OUTPUTS / "resumen_02_carbono_procesos.txt"
    ruta_resumen.write_text(resumen + f"\n  - {ruta_resumen.relative_to(RAIZ)}\n", encoding="utf-8")
    print(resumen)
    print(f"  - {ruta_resumen.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()