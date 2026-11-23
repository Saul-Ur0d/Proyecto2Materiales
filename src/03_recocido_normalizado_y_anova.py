#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_recocido_normalizado_y_anova.py
==================================
Exploración de los registros de recocido y normalizado, diagnóstico de las limitaciones del
dataset y ANOVA (con verificación de supuestos y respaldo no paramétrico).

Entrada
-------
data/processed/aceros_limpio.csv      (generada por el script 01)

Salidas
-------
outputs/tablas/tabla_03_*.csv         diagnóstico, estadísticas, ANOVA/supuestos, post-hoc, pares, tendencias,
                                      atípicos y sensibilidad a anomalías
outputs/resumen_03_recocido_normalizado_anova.txt   resumen interpretado (también se imprime)
outputs/figuras/fig03_01 ... fig03_06 (PNG, 300 dpi)

Diseños de comparación (cada uno recibe un veredicto de validez)
----------------------------------------------------------------
  D1  Annealed vs Normalized                         (térmicos puros; los mismos 20 grados en ambos grupos)
  D2  Annealed vs Spheroidized annealed vs Normalized (ejemplo clásico de 3 grupos)
  D3  Annealed vs Annealed + Cold drawn              (efecto de añadir estirado en frío)
  D4  Normalized vs Normalized + Cold drawn          (n = 2: se documenta como no evaluable)

Lógica estadística (por diseño y variable)
------------------------------------------
  1. Shapiro-Wilk por grupo (normalidad) y Levene/Brown-Forsythe (homogeneidad de varianzas).
  2. ANOVA de un factor (scipy.stats.f_oneway) + eta² y omega².
  3. ANOVA de Welch (respaldo si las varianzas difieren) y Kruskal-Wallis (respaldo no paramétrico).
  4. Prueba recomendada según supuestos y tamaños de muestra; p ajustado por Holm entre las 4 variables.
  5. Veredicto de VALIDEZ del diseño: aunque el ANOVA sea calculable, solo se interpreta como efecto del
     tratamiento si no hay confusión con %C, familia SAE o estirado en frío.
  6. Como D1 es un diseño emparejado (mismo grado en ambos tratamientos), se agrega el análisis pareado
     (t pareada / Wilcoxon), que es la inferencia correcta para esa comparación.

Manejo de anomalías (flag_anomalia, generada por 01)
----------------------------------------------------
  --anomalias resaltar (defecto) | excluir | incluir      --valores corregidos (defecto) | originales
En cualquier modo se exporta la tabla de sensibilidad (con vs sin registros marcados).

Uso
---
    python src/03_recocido_normalizado_y_anova.py
    python src/03_recocido_normalizado_y_anova.py --anomalias excluir
"""

from __future__ import annotations

import argparse
import re
import textwrap
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: funciona en servidores/CI y en VS Code
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy import stats

# ----------------------------------------------------------------------------
# 1. RUTAS
# ----------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[1]
DIR_PROCESSED = RAIZ / "data" / "processed"
DIR_OUTPUTS = RAIZ / "outputs"
DIR_FIGURAS = DIR_OUTPUTS / "figuras"
DIR_TABLAS = DIR_OUTPUTS / "tablas"
ARCHIVO_LIMPIO = DIR_PROCESSED / "aceros_limpio.csv"

# ----------------------------------------------------------------------------
# 2. CONFIGURACIÓN
# ----------------------------------------------------------------------------
VARIABLES = {
    "UTS": "UTS (MPa)",
    "YS": "YS (MPa)",
    "HB": "Dureza Brinell (HB)",
    "ELONG": "Elongación (%)",
}
UNIDAD_CORTA = {"UTS": "MPa", "YS": "MPa", "HB": "HB", "ELONG": "p.p."}  # p.p. = puntos porcentuales

GRUPOS_FOCO = ["Annealed", "Normalized", "Annealed + Cold drawn", "Spheroidized annealed",
               "Normalized + Cold drawn"]                     # orden de presentación: puros primero
GRUPOS_PUROS = ["Annealed", "Normalized"]
COLORES = {
    "Annealed": "#009E73",
    "Normalized": "#D55E00",
    "Annealed + Cold drawn": "#56B4E9",
    "Spheroidized annealed": "#CC79A7",
    "Normalized + Cold drawn": "#8C8C8C",
}
FAMILIAS = {  # familia -> (marcador, etiqueta corta, color)
    "Carbono simple 10xx": ("o", "10xx", "#4C72B0"),
    "Resulfurados/Refosforados 11xx-12xx": ("s", "11xx-12xx", "#DD8452"),
    "Alto Mn 13xx-15xx": ("^", "13xx-15xx", "#55A868"),
    "Aleados 3xxx-9xxx": ("D", "Aleados 3xxx-9xxx", "#C44E52"),
}
ROJO = "#E41A1C"

DISENOS = [
    {"id": "D1", "nombre": "Annealed vs Normalized", "grupos": ["Annealed", "Normalized"]},
    {"id": "D2", "nombre": "Annealed vs Spheroidized annealed vs Normalized",
     "grupos": ["Annealed", "Spheroidized annealed", "Normalized"]},
    {"id": "D3", "nombre": "Annealed vs Annealed + Cold drawn", "grupos": ["Annealed", "Annealed + Cold drawn"]},
    {"id": "D4", "nombre": "Normalized vs Normalized + Cold drawn",
     "grupos": ["Normalized", "Normalized + Cold drawn"]},
]

# Umbrales de diagnóstico
UMBRAL_DIF_C_MEDIO = 0.10     # diferencia de %C medio entre grupos que ya confunde la comparación
UMBRAL_SOLAPE_C = 0.50        # solapamiento mínimo de rangos de %C
UMBRAL_DIF_FAMILIA = 0.25     # distancia de variación total entre composiciones de familia SAE
UMBRAL_DIF_ESTIRADO = 0.25    # diferencia en proporción de registros estirados en frío
RANGO_C_TENDENCIA = 0.15      # rango mínimo de %C para considerar informativa una tendencia
N_MIN_TENDENCIA = 5
N_MIN_ATIPICOS = 6
N_SHAPIRO_POTENCIA = 8        # por debajo de este n, Shapiro tiene muy baja potencia


# ----------------------------------------------------------------------------
# 3. TEXTOS METALÚRGICOS Y DE INTERPRETACIÓN
# ----------------------------------------------------------------------------
TEXTO_METALURGICO = [
    "Recocido (Annealed): calentamiento a austenización y enfriamiento lento en horno; la transformación "
    "ocurre cerca del equilibrio y da perlita gruesa (mayor espaciamiento interlaminar) y grano más grande: "
    "menor resistencia y dureza, mayor ductilidad.",
    "Normalizado (Normalized): austenización y enfriamiento al aire, más rápido que en horno; produce perlita "
    "más fina y grano austenítico refinado y homogéneo: más UTS, YS y dureza, con menos elongación que el "
    "recocido de la misma composición.",
    "Aceros de alta templabilidad (4340, 4150, 4140, 8650...): con Cr, Mo y Ni la curva de enfriamiento al aire "
    "puede cruzar el campo de bainita (e incluso martensita parcial). Por eso el salto al normalizar puede ser "
    "mucho mayor que el esperable solo por perlita fina: la diferencia depende del grado.",
    "Esferoidizado (Spheroidized annealed): la cementita se transforma de láminas a glóbulos en matriz ferrítica; "
    "estructura blanda y de buena maquinabilidad. En este dataset todos sus registros están además estirados en "
    "frío (endurecimiento por deformación: ↑ dislocaciones, ↑ YS, ↓ elongación), así que no representan la "
    "microestructura esferoidizada pura.",
    "Estirado en frío (Cold drawn): se añade a una base recocida o normalizada y eleva YS y UTS a costa de "
    "elongación; por eso 'Annealed + Cold drawn' y 'Normalized + Cold drawn' no son comparables con los "
    "tratamientos térmicos puros.",
]

GUIA_ANOVA = [
    "Supuestos del ANOVA de un factor: (1) observaciones independientes, (2) residuos aproximadamente normales y "
    "(3) varianzas homogéneas entre grupos.",
    "Shapiro-Wilk con n pequeño (< 8) tiene muy poca potencia: 'no rechazar la normalidad' no la demuestra. "
    "Con grupos balanceados y n ≥ 15-20 el ANOVA es robusto a desviaciones moderadas de normalidad; con grupos "
    "desiguales o varianzas distintas conviene Welch.",
    "Kruskal-Wallis (respaldo) no supone normalidad, pero compara rangos/distribuciones y no medias, y también "
    "exige independencia.",
    "El valor p depende del tamaño de muestra: con n pequeño un efecto grande puede no ser significativo y con "
    "n grande uno trivial sí; por eso se reporta eta² (≥ 0.01 pequeño, ≥ 0.06 mediano, ≥ 0.14 grande).",
    "Aquí cada renglón es un valor nominal de tabla SAE de un grado distinto (no una probeta): los grados "
    "vecinos son parecidos entre sí, de modo que la independencia se cumple solo de forma aproximada y los "
    "valores p tienden a ser optimistas.",
    "Se ajusta por Holm entre las 4 variables mecánicas de cada diseño (comparaciones múltiples).",
    "Un ANOVA calculable no equivale a un ANOVA interpretable: si los grupos difieren también en %C, familia SAE "
    "o estirado en frío, la diferencia observada no puede atribuirse al tratamiento térmico.",
]


# ----------------------------------------------------------------------------
# 4. ARGUMENTOS Y CARGA DE DATOS
# ----------------------------------------------------------------------------
def leer_argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recocido, normalizado y ANOVA.")
    p.add_argument("--anomalias", choices=["resaltar", "excluir", "incluir"], default="resaltar",
                   help="Tratamiento de registros con flag_anomalia (defecto: resaltar).")
    p.add_argument("--valores", choices=["corregidos", "originales"], default="corregidos",
                   help="UTS/YS conciliados (UTS_MPa_corr, YS_MPa_corr) u originales (defecto: corregidos).")
    p.add_argument("--alfa", type=float, default=0.05, help="Nivel de significancia (defecto: 0.05).")
    p.add_argument("--min-n", type=int, default=3,
                   help="n mínimo por grupo para ejecutar las pruebas (defecto: 3).")
    p.add_argument("--dpi", type=int, default=300, help="Resolución de las figuras (defecto: 300).")
    return p.parse_args()


def cargar_datos() -> pd.DataFrame:
    if not ARCHIVO_LIMPIO.exists():
        raise FileNotFoundError(
            f"No existe {ARCHIVO_LIMPIO.relative_to(RAIZ)}. Ejecuta primero: python src/01_limpieza_datos.py")
    df = pd.read_csv(ARCHIVO_LIMPIO, encoding="utf-8-sig")
    requeridas = ["registro", "SAE Grade", "Familia_SAE", "Conditions_limpia", "Condicion_Grupo", "Temp_C",
                  "C_pct", "UTS (MPa)", "YS (MPa)", "UTS_MPa_corr", "YS_MPa_corr", "Hardness_HB_num",
                  "Elongation (%)", "flag_anomalia", "motivo_auditoria"]
    faltan = [c for c in requeridas if c not in df.columns]
    if faltan:
        raise KeyError(f"Faltan columnas en aceros_limpio.csv: {faltan}. ¿Se ejecutó la versión actual del script 01?")
    df["flag_anomalia"] = df["flag_anomalia"].astype(str).str.lower().eq("true")
    df["motivo_auditoria"] = df["motivo_auditoria"].fillna("")
    df["SAE Grade"] = df["SAE Grade"].astype(str)
    return df


def preparar_variables(df: pd.DataFrame, usar_corregidos: bool) -> pd.DataFrame:
    """Columnas de análisis (UTS, YS, HB, ELONG) y etiquetas de tratamiento."""
    d = df.copy()
    d["UTS"] = d["UTS_MPa_corr"] if usar_corregidos else d["UTS (MPa)"]
    d["YS"] = d["YS_MPa_corr"] if usar_corregidos else d["YS (MPa)"]
    d["HB"] = d["Hardness_HB_num"]
    d["ELONG"] = d["Elongation (%)"]
    d["Estirado_en_frio"] = d["Conditions_limpia"].fillna("").str.lower().str.contains("cold drawn")
    d["Tratamiento_Termico"] = np.select(
        [d["Condicion_Grupo"].str.contains("nnealed"), d["Condicion_Grupo"].str.contains("Normalized")],
        ["Recocido", "Normalizado"], default="Otro")
    d["Tipo_Proceso"] = np.where(d["Estirado_en_frio"], "Térmico + estirado en frío", "Térmico puro")
    return d


def grupos_presentes(d: pd.DataFrame) -> list[str]:
    existentes = set(d["Condicion_Grupo"])
    return [g for g in GRUPOS_FOCO if g in existentes]


# ----------------------------------------------------------------------------
# 5. FUNCIONES ESTADÍSTICAS
# ----------------------------------------------------------------------------
def p_texto(p) -> str:
    if p is None or pd.isna(p):
        return "NA"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def num(x, f: str = ".2f") -> str:
    return "NA" if x is None or pd.isna(x) else format(x, f)


def magnitud_efecto(e) -> str:
    if e is None or pd.isna(e):
        return "NA"
    return "grande" if e >= 0.14 else "mediano" if e >= 0.06 else "pequeño" if e >= 0.01 else "despreciable"


def ajuste_holm(p) -> np.ndarray:
    """Corrección de Holm-Bonferroni (ignora NaN)."""
    p = np.asarray(p, dtype=float)
    salida = np.full_like(p, np.nan)
    idx = np.where(np.isfinite(p))[0]
    if len(idx) == 0:
        return salida
    orden = idx[np.argsort(p[idx])]
    m, previo = len(orden), 0.0
    for rango, i in enumerate(orden):
        previo = max(previo, min(1.0, (m - rango) * p[i]))
        salida[i] = previo
    return salida


def prueba_shapiro(x) -> tuple[float, str]:
    """Shapiro-Wilk; devuelve (p, motivo si no es calculable)."""
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return np.nan, "n<3"
    if np.ptp(x) == 0:
        return np.nan, "valores constantes"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(stats.shapiro(x).pvalue), ""


def prueba_levene(arrays: list[np.ndarray]) -> float:
    """Levene con centro en la mediana (Brown-Forsythe, robusto a no normalidad)."""
    if len(arrays) < 2 or any(len(a) < 2 for a in arrays) or all(np.ptp(a) == 0 for a in arrays):
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(stats.levene(*arrays, center="median").pvalue)


def anova_clasico(arrays: list[np.ndarray]) -> dict:
    """ANOVA de un factor (f_oneway) + eta² y omega²."""
    k, N = len(arrays), sum(len(a) for a in arrays)
    todo = np.concatenate(arrays)
    vacio = {"F": np.nan, "df1": k - 1, "df2": N - k, "p": np.nan, "eta2": np.nan, "omega2": np.nan}
    if np.ptp(todo) == 0 or N - k <= 0:
        return vacio
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        F, p = stats.f_oneway(*arrays)
    gm = todo.mean()
    ss_b = sum(len(a) * (a.mean() - gm) ** 2 for a in arrays)
    ss_t = ((todo - gm) ** 2).sum()
    ms_w = (ss_t - ss_b) / (N - k)
    omega = (ss_b - (k - 1) * ms_w) / (ss_t + ms_w)
    return {"F": float(F), "df1": k - 1, "df2": N - k, "p": float(p), "eta2": float(ss_b / ss_t),
            "omega2": float(max(omega, 0.0))}


def anova_welch(arrays: list[np.ndarray]) -> dict:
    """ANOVA de Welch (no supone varianzas iguales). Requiere varianza > 0 en todos los grupos."""
    k = len(arrays)
    n = np.array([len(a) for a in arrays], dtype=float)
    if (n < 2).any() or any(np.ptp(a) == 0 for a in arrays):
        return {"F": np.nan, "df2": np.nan, "p": np.nan}
    m = np.array([a.mean() for a in arrays])
    v = np.array([a.var(ddof=1) for a in arrays])
    w = n / v
    W = w.sum()
    mw = (w * m).sum() / W
    lam = ((1 - w / W) ** 2 / (n - 1)).sum()
    F = ((w * (m - mw) ** 2).sum() / (k - 1)) / (1 + 2 * (k - 2) / (k ** 2 - 1) * lam)
    df2 = (k ** 2 - 1) / (3 * lam)
    return {"F": float(F), "df2": float(df2), "p": float(stats.f.sf(F, k - 1, df2))}


def kruskal_wallis(arrays: list[np.ndarray]) -> dict:
    """Kruskal-Wallis con épsilon² como tamaño del efecto (H / (N-1))."""
    N = sum(len(a) for a in arrays)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            H, p = stats.kruskal(*arrays)
    except ValueError:  # todos los valores idénticos
        return {"H": np.nan, "p": np.nan, "eps2": np.nan}
    return {"H": float(H), "p": float(p), "eps2": float(H / (N - 1))}


def posthoc_tukey(nombres: list[str], arrays: list[np.ndarray]) -> list[dict]:
    """Tukey HSD (scipy) para pares de grupos; devuelve filas."""
    if any(len(a) < 2 for a in arrays) or all(np.ptp(a) == 0 for a in arrays):
        return []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = stats.tukey_hsd(*arrays)
            ic = res.confidence_interval()
    except Exception:  # noqa: BLE001 - se omite el post-hoc si scipy no puede calcularlo
        return []
    filas = []
    for i, j in combinations(range(len(arrays)), 2):
        filas.append({"metodo": "Tukey HSD", "grupo_a": nombres[i], "grupo_b": nombres[j],
                      "diferencia": float(arrays[i].mean() - arrays[j].mean()),
                      "ic95_inf": float(ic.low[i, j]), "ic95_sup": float(ic.high[i, j]),
                      "p_sin_ajustar": np.nan, "p_ajustado": float(res.pvalue[i, j])})
    return filas


def posthoc_mann_whitney(nombres: list[str], arrays: list[np.ndarray]) -> list[dict]:
    """Mann-Whitney por pares con ajuste de Holm. diferencia = mediana_a - mediana_b."""
    filas = []
    for i, j in combinations(range(len(arrays)), 2):
        a, b = arrays[i], arrays[j]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = stats.mannwhitneyu(a, b, alternative="two-sided")
            p = float(res.pvalue)
            rb = float(2 * res.statistic / (len(a) * len(b)) - 1)   # correlación biserial por rangos
        except ValueError:
            p, rb = np.nan, np.nan
        filas.append({"metodo": "Mann-Whitney + Holm", "grupo_a": nombres[i], "grupo_b": nombres[j],
                      "diferencia": float(np.median(a) - np.median(b)), "ic95_inf": np.nan, "ic95_sup": np.nan,
                      "p_sin_ajustar": p, "p_ajustado": np.nan, "biserial_rangos": rb})
    ajustados = ajuste_holm([f["p_sin_ajustar"] for f in filas])
    for f, pa in zip(filas, ajustados):
        f["p_ajustado"] = float(pa)
    return filas


def estadisticos_relacion(x: pd.Series, y: pd.Series) -> dict:
    """Pearson, Spearman y regresión lineal simple y = a + b·x (con IC 95 % de la pendiente)."""
    vacio = {k: np.nan for k in ("pearson_r", "pearson_p", "spearman_rho", "spearman_p", "pendiente",
                                 "pendiente_por_0_1C", "intercepto", "r2")}
    m = x.notna() & y.notna()
    x, y = x[m].astype(float), y[m].astype(float)
    n = len(x)
    if n < 3 or x.nunique() < 2 or y.nunique() < 2:
        return {"n": n, **vacio}
    r, pr = stats.pearsonr(x, y)
    rho, ps = stats.spearmanr(x, y)
    lr = stats.linregress(x, y)
    return {"n": n, "pearson_r": r, "pearson_p": pr, "spearman_rho": rho, "spearman_p": ps,
            "pendiente": lr.slope, "pendiente_por_0_1C": lr.slope * 0.1, "intercepto": lr.intercept,
            "r2": lr.rvalue ** 2}


# ----------------------------------------------------------------------------
# 6. DIAGNÓSTICO DE LIMITACIONES
# ----------------------------------------------------------------------------
def distancia_variacion_total(a: pd.Series, b: pd.Series) -> float:
    """Distancia entre dos distribuciones de categorías (0 = idénticas, 1 = disjuntas)."""
    pa, pb = a.value_counts(normalize=True), b.value_counts(normalize=True)
    cats = set(pa.index) | set(pb.index)
    return float(0.5 * sum(abs(pa.get(c, 0.0) - pb.get(c, 0.0)) for c in cats))


def diagnostico_diseno(d: pd.DataFrame, grupos: list[str]) -> dict:
    """Evalúa si la comparación entre grupos puede atribuirse al tratamiento o está confundida."""
    sub = {g: d[d["Condicion_Grupo"] == g] for g in grupos}
    ns = {g: len(s) for g, s in sub.items()}
    medias_c = {g: s["C_pct"].mean() for g, s in sub.items()}
    rango_medias_c = (max(medias_c.values()) - min(medias_c.values())) if all(ns.values()) else np.nan

    solapes, dist_fam = [], []
    for a, b in combinations(grupos, 2):
        if ns[a] == 0 or ns[b] == 0:
            continue
        lo = max(sub[a]["C_pct"].min(), sub[b]["C_pct"].min())
        hi = min(sub[a]["C_pct"].max(), sub[b]["C_pct"].max())
        amplitud = max(sub[a]["C_pct"].max(), sub[b]["C_pct"].max()) - min(sub[a]["C_pct"].min(), sub[b]["C_pct"].min())
        solapes.append(max(0.0, hi - lo) / amplitud if amplitud > 0 else 1.0)
        dist_fam.append(distancia_variacion_total(sub[a]["Familia_SAE"], sub[b]["Familia_SAE"]))
    solape_min = min(solapes) if solapes else np.nan
    dif_familia = max(dist_fam) if dist_fam else np.nan

    estirado = {g: float(s["Estirado_en_frio"].mean()) if len(s) else np.nan for g, s in sub.items()}
    dif_estirado = np.nanmax(list(estirado.values())) - np.nanmin(list(estirado.values()))

    conjuntos = [set(s["SAE Grade"]) for s in sub.values() if len(s)]
    union = set.union(*conjuntos) if conjuntos else set()
    pareado = (len(set.intersection(*conjuntos)) / len(union)) if union else 0.0
    temps = {g: s["Temp_C"].mean() for g, s in sub.items()}

    confusion = []
    if pd.notna(rango_medias_c) and rango_medias_c > UMBRAL_DIF_C_MEDIO:
        confusion.append(f"%C medio distinto entre grupos (diferencia máxima {rango_medias_c:.2f} %C)")
    if pd.notna(solape_min) and solape_min < UMBRAL_SOLAPE_C:
        confusion.append(f"rangos de %C casi sin solapamiento (solapamiento {solape_min:.0%}): no hay soporte común")
    if pd.notna(dif_familia) and dif_familia > UMBRAL_DIF_FAMILIA:
        confusion.append(f"composición por familia SAE distinta entre grupos (distancia {dif_familia:.2f})")
    if pd.notna(dif_estirado) and dif_estirado > UMBRAL_DIF_ESTIRADO:
        confusion.append("mezcla de tratamiento térmico puro con estirado en frío entre grupos")

    return {"n": ns, "medias_c": medias_c, "rango_medias_c": rango_medias_c, "solape_c_min": solape_min,
            "dif_familia": dif_familia, "estirado": estirado, "dif_estirado": dif_estirado,
            "pareado": pareado, "temp_media": temps, "confusion": confusion,
            "desbalance": (max(ns.values()) / min(ns.values())) if min(ns.values()) else np.inf}


def veredicto_validez(diag: dict, min_n: int) -> tuple[str, list[str]]:
    """Nivel de validez del diseño y razones."""
    razones: list[str] = []
    n_min = min(diag["n"].values())
    if n_min < min_n:
        return "NO EVALUABLE", [f"algún grupo tiene n < {min_n} (n mínimo = {n_min})"]
    if diag["confusion"]:
        return "NO VÁLIDO PARA ATRIBUIR EL EFECTO AL TRATAMIENTO", diag["confusion"]
    if diag["pareado"] >= 0.9:
        razones.append("diseño emparejado (mismos grados en ambos grupos): el ANOVA independiente ignora el "
                       "emparejamiento; la inferencia correcta es la prueba pareada")
    if diag["desbalance"] > 1.5:
        razones.append(f"grupos desbalanceados (razón de tamaños {diag['desbalance']:.1f})")
    if n_min < N_SHAPIRO_POTENCIA:
        razones.append("n pequeño: baja potencia en las pruebas de supuestos")
    razones.append("valores nominales de tabla, un renglón por grado (independencia aproximada)")
    return "VÁLIDO CON RESERVAS", razones


def tabla_diagnostico_grupos(d: pd.DataFrame, atipicos: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for g in grupos_presentes(d):
        s = d[d["Condicion_Grupo"] == g]
        fam = s["Familia_SAE"].value_counts()
        sin_var = [VARIABLES[k] for k in VARIABLES if s[k].dropna().nunique() == 1 and s[k].notna().sum() > 1]
        filas.append({
            "Condicion_Grupo": g, "n_registros": len(s), "n_grados": s["SAE Grade"].nunique(),
            "Tipo_Proceso": "Térmico + estirado en frío" if s["Estirado_en_frio"].mean() > 0.5 else "Térmico puro",
            "pct_estirado_en_frio": round(100 * s["Estirado_en_frio"].mean(), 1),
            "familias_SAE": "; ".join(f"{k}: {v}" for k, v in fam.items()),
            "pct_aleados": round(100 * (s["Familia_SAE"] == "Aleados 3xxx-9xxx").mean(), 1),
            "C_pct_media": round(s["C_pct"].mean(), 3), "C_pct_min": s["C_pct"].min(), "C_pct_max": s["C_pct"].max(),
            "Temp_C_media": round(s["Temp_C"].mean(), 1) if s["Temp_C"].notna().any() else np.nan,
            "n_flag_anomalia": int(s["flag_anomalia"].sum()),
            "n_atipicos_IQR": int((atipicos["Condicion_Grupo"] == g).sum()) if not atipicos.empty else 0,
            "variables_sin_variabilidad": "; ".join(sin_var),
            "condicion_mas_frecuente": s["Conditions_limpia"].mode().iloc[0] if len(s) else "",
        })
    return pd.DataFrame(filas)


def detectar_atipicos(d: pd.DataFrame, grupos: list[str]) -> pd.DataFrame:
    """Atípicos por regla 1.5×IQR dentro de cada grupo y variable (solo grupos con n ≥ N_MIN_ATIPICOS)."""
    filas = []
    for g in grupos:
        for clave, etiqueta in VARIABLES.items():
            s = d[d["Condicion_Grupo"] == g].dropna(subset=[clave])
            if len(s) < N_MIN_ATIPICOS:
                continue
            q1, q3 = s[clave].quantile([0.25, 0.75])
            lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
            for _, f in s[(s[clave] < lo) | (s[clave] > hi)].iterrows():
                filas.append({"Condicion_Grupo": g, "variable": etiqueta, "clave": clave, "registro": int(f["registro"]),
                              "SAE Grade": f["SAE Grade"], "valor": f[clave], "limite_inferior": lo,
                              "limite_superior": hi, "flag_anomalia": bool(f["flag_anomalia"]),
                              "categorias_auditoria": ", ".join(sorted(set(re.findall(
                                  r"\[ANOMALIA:([A-Z_]+)\]", f["motivo_auditoria"]))))})
    return pd.DataFrame(filas, columns=["Condicion_Grupo", "variable", "clave", "registro", "SAE Grade", "valor",
                                        "limite_inferior", "limite_superior", "flag_anomalia", "categorias_auditoria"])


def estadisticas_grupos(d: pd.DataFrame) -> pd.DataFrame:
    """n, media, mediana, desv. est., mín, máx y nº de valores únicos por grupo y variable."""
    variables = {**VARIABLES, "C_pct": "%C (C_pct)", "Temp_C": "Temperatura de tratamiento (°C)"}
    filas = []
    for g in grupos_presentes(d):
        s0 = d[d["Condicion_Grupo"] == g]
        for clave, etiqueta in variables.items():
            s = s0[clave].dropna()
            filas.append({"Condicion_Grupo": g, "variable": etiqueta, "n": len(s),
                          "media": s.mean() if len(s) else np.nan, "mediana": s.median() if len(s) else np.nan,
                          "desv_est": s.std(ddof=1) if len(s) > 1 else np.nan,
                          "min": s.min() if len(s) else np.nan, "max": s.max() if len(s) else np.nan,
                          "n_valores_unicos": int(s.nunique())})
    return pd.DataFrame(filas)


def tendencias_c(d: pd.DataFrame) -> pd.DataFrame:
    """C_pct vs cada propiedad, dentro de cada grupo."""
    filas = []
    for g in grupos_presentes(d):
        s = d[d["Condicion_Grupo"] == g]
        rango = s["C_pct"].max() - s["C_pct"].min()
        for clave, etiqueta in VARIABLES.items():
            e = estadisticos_relacion(s["C_pct"], s[clave])
            notas = []
            if e["n"] < N_MIN_TENDENCIA:
                notas.append(f"n < {N_MIN_TENDENCIA}")
            if rango < RANGO_C_TENDENCIA - 1e-9:
                notas.append(f"rango de %C estrecho (< {RANGO_C_TENDENCIA} %)")
            if s[clave].dropna().nunique() <= 1:
                notas.append("propiedad constante")
            filas.append({"Condicion_Grupo": g, "clave": clave, "variable": etiqueta, "rango_C": round(rango, 3),
                          **e, "nota": "; ".join(notas)})
    return pd.DataFrame(filas)


# ----------------------------------------------------------------------------
# 7. ANOVA POR DISEÑO
# ----------------------------------------------------------------------------
def probar_variable(d: pd.DataFrame, grupos: list[str], clave: str, etiqueta: str, alfa: float, min_n: int) -> dict:
    """Supuestos + ANOVA + Welch + Kruskal-Wallis para una variable y un conjunto de grupos."""
    arrays = {g: d.loc[d["Condicion_Grupo"] == g, clave].dropna().to_numpy(dtype=float) for g in grupos}
    ns = {g: len(a) for g, a in arrays.items()}
    reg = {"variable": etiqueta, "clave": clave, "k_grupos": len(grupos), "N": sum(ns.values()),
           "n_por_grupo": "; ".join(f"{g}={n}" for g, n in ns.items()), "n_min": min(ns.values())}
    if min(ns.values()) < min_n:
        reg["prueba_recomendada"] = "No evaluable"
        return reg
    lista = [arrays[g] for g in grupos]

    # Supuestos
    shap, no_verificables = {}, []
    for g in grupos:
        p, motivo = prueba_shapiro(arrays[g])
        shap[g] = p
        if motivo:
            no_verificables.append(f"{g} ({motivo})")
    reg["shapiro_p_por_grupo"] = "; ".join(f"{g}: {p_texto(p)}" for g, p in shap.items())
    reg["shapiro_no_verificable"] = "; ".join(no_verificables)
    p_validos = [p for p in shap.values() if pd.notna(p)]
    if no_verificables:
        normalidad = None                                  # no se puede verificar en todos los grupos
    else:
        normalidad = all(p >= alfa for p in p_validos)
    reg["normalidad_ok"] = normalidad
    reg["levene_p"] = prueba_levene(lista)
    varianzas_ok = bool(pd.notna(reg["levene_p"]) and reg["levene_p"] >= alfa)
    reg["varianzas_ok"] = varianzas_ok if pd.notna(reg["levene_p"]) else None

    # Pruebas
    a = anova_clasico(lista)
    w = anova_welch(lista)
    k = kruskal_wallis(lista)
    reg.update({"anova_F": a["F"], "anova_df1": a["df1"], "anova_df2": a["df2"], "anova_p": a["p"],
                "eta2": a["eta2"], "omega2": a["omega2"], "efecto_eta2": magnitud_efecto(a["eta2"]),
                "welch_F": w["F"], "welch_df2": w["df2"], "welch_p": w["p"],
                "kw_H": k["H"], "kw_p": k["p"], "kw_eps2": k["eps2"]})

    # Prueba recomendada según supuestos
    if normalidad is True and varianzas_ok:
        reg["prueba_recomendada"], reg["p_recomendado"] = "ANOVA clásico", a["p"]
    elif normalidad is True and pd.notna(w["p"]):
        reg["prueba_recomendada"], reg["p_recomendado"] = "ANOVA de Welch", w["p"]
    else:
        reg["prueba_recomendada"], reg["p_recomendado"] = "Kruskal-Wallis", k["p"]
    reg["concordancia_anova_kw"] = (bool(a["p"] < alfa) == bool(k["p"] < alfa)) if pd.notna(a["p"]) and pd.notna(k["p"]) else None
    reg["_arrays"] = arrays
    return reg


def probar_diseno(d: pd.DataFrame, diseno: dict, alfa: float, min_n: int) -> tuple[list[dict], list[dict], dict, str, list[str]]:
    """Ejecuta las pruebas de un diseño (4 variables) y los post-hoc si k ≥ 3."""
    grupos = [g for g in diseno["grupos"] if (d["Condicion_Grupo"] == g).any()]
    diag = diagnostico_diseno(d, diseno["grupos"])
    nivel, razones = veredicto_validez(diag, min_n)
    regs = [probar_variable(d, diseno["grupos"], c, e, alfa, min_n) for c, e in VARIABLES.items()]

    holm = ajuste_holm([r.get("p_recomendado", np.nan) for r in regs])
    posthoc = []
    for r, ph in zip(regs, holm):
        r["p_holm"] = ph
        r["diseno_id"], r["diseno"] = diseno["id"], diseno["nombre"]
        r["nivel_validez"], r["razones_validez"] = nivel, " | ".join(razones)
        if len(grupos) >= 3 and r.get("prueba_recomendada") not in (None, "No evaluable"):
            nombres = diseno["grupos"]
            lista = [r["_arrays"][g] for g in nombres]
            for fila in posthoc_tukey(nombres, lista) + posthoc_mann_whitney(nombres, lista):
                fila.update({"diseno_id": diseno["id"], "variable": r["variable"],
                             "recomendado": (fila["metodo"] == "Tukey HSD") == (r["prueba_recomendada"] != "Kruskal-Wallis")})
                posthoc.append(fila)
    return regs, posthoc, diag, nivel, razones


# ----------------------------------------------------------------------------
# 8. ANÁLISIS PAREADO (mismo SAE Grade en ambos tratamientos)
# ----------------------------------------------------------------------------
def analisis_pareado(d: pd.DataFrame, g_a: str, g_b: str, alfa: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Diferencias g_b - g_a por grado (composición nominal idéntica) y su prueba pareada."""
    a = d[d["Condicion_Grupo"] == g_a].drop_duplicates("SAE Grade").set_index("SAE Grade")
    b = d[d["Condicion_Grupo"] == g_b].drop_duplicates("SAE Grade").set_index("SAE Grade")
    grados = sorted(set(a.index) & set(b.index))
    ta, tb = g_a.replace(" ", ""), g_b.replace(" ", "")

    filas = []
    for g in grados:
        fila = {"SAE Grade": g, "Familia_SAE": a.at[g, "Familia_SAE"], "C_pct": a.at[g, "C_pct"],
                f"Temp_C_{ta}": a.at[g, "Temp_C"], f"Temp_C_{tb}": b.at[g, "Temp_C"],
                "delta_Temp_C": b.at[g, "Temp_C"] - a.at[g, "Temp_C"],
                "flag_anomalia": bool(a.at[g, "flag_anomalia"] or b.at[g, "flag_anomalia"])}
        for clave in VARIABLES:
            fila[f"{clave}_{ta}"], fila[f"{clave}_{tb}"] = a.at[g, clave], b.at[g, clave]
            fila[f"delta_{clave}"] = b.at[g, clave] - a.at[g, clave]
        filas.append(fila)
    detalle = pd.DataFrame(filas)
    if detalle.empty:
        return detalle, pd.DataFrame()

    resumen = []
    for clave, etiqueta in VARIABLES.items():
        x, y = detalle[f"{clave}_{ta}"], detalle[f"{clave}_{tb}"]
        ok = x.notna() & y.notna()
        delta = (y - x)[ok].to_numpy(dtype=float)
        n = len(delta)
        if n < 3:
            resumen.append({"variable": etiqueta, "clave": clave, "n_pares": n})
            continue
        sd = delta.std(ddof=1)
        ic = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
        p_sh, _ = prueba_shapiro(delta)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p_t = float(stats.ttest_rel(y[ok], x[ok]).pvalue)
            try:
                p_w = float(stats.wilcoxon(delta).pvalue)
            except ValueError:
                p_w = np.nan
        usa_t = pd.notna(p_sh) and p_sh >= alfa
        rho = stats.spearmanr(detalle.loc[ok, "C_pct"], delta) if n >= 3 and np.ptp(delta) > 0 else None
        resumen.append({
            "variable": etiqueta, "clave": clave, "n_pares": n, "delta_medio": delta.mean(),
            "delta_mediana": float(np.median(delta)), "delta_desv_est": sd,
            "ic95_inf": delta.mean() - ic, "ic95_sup": delta.mean() + ic,
            "d_cohen_dz": delta.mean() / sd if sd > 0 else np.nan,
            "delta_relativo_medio_pct": float(((y - x) / x * 100)[ok].mean()),
            "pares_con_aumento": int((delta > 0).sum()), "pares_con_descenso": int((delta < 0).sum()),
            "shapiro_p_diferencias": p_sh, "p_ttest_pareado": p_t, "p_wilcoxon": p_w,
            "prueba_recomendada": "t pareada" if usa_t else "Wilcoxon",
            "p_recomendado": p_t if usa_t else p_w,
            "rho_delta_vs_C": float(rho.statistic) if rho is not None else np.nan,
            "p_rho_delta_vs_C": float(rho.pvalue) if rho is not None else np.nan,
        })
    res = pd.DataFrame(resumen)
    if "p_recomendado" in res:
        res["p_holm"] = ajuste_holm(res["p_recomendado"].to_numpy(dtype=float))
    return detalle, res


# ----------------------------------------------------------------------------
# 9. SENSIBILIDAD A ANOMALÍAS
# ----------------------------------------------------------------------------
def tabla_sensibilidad(foco_completo: pd.DataFrame, alfa: float, min_n: int) -> pd.DataFrame:
    """D1 (ANOVA independiente y pareado) con todos los registros vs sin los marcados."""
    filas = []
    conjuntos = {"todos": foco_completo, "sin_anomalias": foco_completo[~foco_completo["flag_anomalia"]]}
    for nombre, datos in conjuntos.items():
        regs, _, _, _, _ = probar_diseno(datos, DISENOS[0], alfa, min_n)
        _, par = analisis_pareado(datos, "Annealed", "Normalized", alfa)
        for r in regs:
            p = par[par["clave"] == r["clave"]]
            fila = {"conjunto": nombre, "variable": r["variable"], "n_registros": int(r["N"]),
                    "anova_F": r.get("anova_F", np.nan), "anova_p": r.get("anova_p", np.nan),
                    "eta2": r.get("eta2", np.nan), "kw_p": r.get("kw_p", np.nan)}
            if not p.empty and "delta_medio" in p:
                fila.update({"n_pares": int(p.iloc[0]["n_pares"]), "delta_medio_pareado": p.iloc[0].get("delta_medio", np.nan),
                             "p_pareado": p.iloc[0].get("p_recomendado", np.nan)})
            filas.append(fila)
    return pd.DataFrame(filas)


# ----------------------------------------------------------------------------
# 10. UTILIDADES GRÁFICAS
# ----------------------------------------------------------------------------
def configurar_estilo() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"axes.titlesize": 11, "axes.labelsize": 11, "legend.fontsize": 9,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})


def pie_de_figura(fig: plt.Figure, texto: str, ancho: int = 165, y: float = 0.012) -> None:
    fig.text(0.5, y, textwrap.fill(texto, ancho), ha="center", va="bottom", fontsize=8.6,
             color="#333333", style="italic")


def guardar_figura(fig: plt.Figure, nombre: str, dpi: int) -> Path:
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    ruta = DIR_FIGURAS / nombre
    fig.savefig(ruta, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return ruta


def etiqueta_modo(modo: str, n_excl: int) -> str:
    if modo == "excluir":
        return f"Se excluyeron {n_excl} registros con flag_anomalia. "
    if modo == "resaltar":
        return "Aro rojo = registro con flag_anomalia. "
    return ""


def fig_diagnostico(d: pd.DataFrame, diag_grupos: pd.DataFrame, dpi: int) -> Path:
    """Figura 1: tamaños de muestra, rango de %C y composición por familia en cada grupo."""
    grupos = grupos_presentes(d)
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.6), gridspec_kw={"width_ratios": [1, 1.15, 1.15]})

    ax = axes[0]
    for i, g in enumerate(grupos):
        s = d[d["Condicion_Grupo"] == g]
        combinado = s["Estirado_en_frio"].mean() > 0.5
        ax.barh(i, len(s), color=COLORES[g], hatch="//" if combinado else None, edgecolor="#444444", alpha=0.85)
        ax.text(len(s) + 0.4, i, f"n = {len(s)}", va="center", fontsize=10)
    ax.set_yticks(range(len(grupos)))
    ax.set_yticklabels(grupos, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, max(len(d[d['Condicion_Grupo'] == g]) for g in grupos) * 1.25)
    ax.set_xlabel("Número de registros")
    ax.set_title("(a) Tamaño de muestra desigual\n(rayado = incluye estirado en frío)", loc="left")

    ax = axes[1]
    rng = np.random.default_rng(7)
    for i, g in enumerate(grupos):
        s = d[d["Condicion_Grupo"] == g]["C_pct"].dropna()
        ax.plot([s.min(), s.max()], [i, i], color=COLORES[g], lw=6, alpha=0.35, solid_capstyle="round")
        ax.scatter(s, i + rng.uniform(-0.12, 0.12, len(s)), c=COLORES[g], s=26, alpha=0.9, edgecolors="white",
                   linewidths=0.4, zorder=3)
        ax.scatter([s.mean()], [i], marker="D", s=70, c="white", edgecolors="black", linewidths=1.2, zorder=5)
    ax.set_yticks(range(len(grupos)))
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlabel("Carbono, %C (promedio de límites Min-Max, % en masa)")
    ax.set_title("(b) Rango de %C por grupo (rombo = media):\nsin soporte común entre algunos grupos", loc="left")

    ax = axes[2]
    tabla = pd.crosstab(d["Condicion_Grupo"], d["Familia_SAE"]).reindex(grupos).fillna(0)
    izq = np.zeros(len(grupos))
    for fam, (_, lab, color) in FAMILIAS.items():
        if fam in tabla.columns:
            ax.barh(range(len(grupos)), tabla[fam].values, left=izq, color=color, label=lab, edgecolor="white")
            izq += tabla[fam].values
    ax.set_yticks(range(len(grupos)))
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlabel("Número de registros")
    ax.set_title("(c) Familias SAE: los grupos térmicos puros son aleados,\nlos combinados son 10xx-15xx", loc="left")
    ax.legend(title="Familia SAE", loc="lower right")

    fig.suptitle("Diagnóstico de limitaciones: recocido y normalizado", fontsize=14, fontweight="bold", y=1.0)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.82, bottom=0.2, wspace=0.08)
    pie_de_figura(fig, "Los grupos no son comparables entre sí sin cautela: difieren en tamaño (n), en %C (Spheroidized annealed "
                       "es de %C alto), en familia SAE (Annealed y Normalized son casi solo aleados) y en si incluyen estirado en "
                       "frío (Spheroidized annealed, Annealed + Cold drawn y Normalized + Cold drawn). Solo Annealed y Normalized "
                       "comparten los mismos grados.", ancho=155)
    return guardar_figura(fig, "fig03_01_diagnostico_limitaciones.png", dpi)


def fig_boxplots(d: pd.DataFrame, atipicos: pd.DataFrame, resaltar: bool, modo: str, n_excl: int, dpi: int) -> Path:
    """Figura 2: distribuciones por grupo (caja + puntos por familia + media), puros vs combinados."""
    posiciones, x = {}, 0.0
    for g in [g for g in GRUPOS_FOCO if g in set(d["Condicion_Grupo"])]:
        if g not in GRUPOS_PUROS and x in (0.0, 1.0, 2.0) or (g not in GRUPOS_PUROS and not posiciones.get("_gap")):
            if g not in GRUPOS_PUROS and not posiciones.get("_gap"):
                x += 0.8
                posiciones["_gap"] = x
        posiciones[g] = x
        x += 1.0
    rng = np.random.default_rng(42)
    grupos = [g for g in posiciones if g != "_gap"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12.5))
    for ax, (clave, etiqueta) in zip(axes.ravel(), VARIABLES.items()):
        for g in grupos:
            s = d[d["Condicion_Grupo"] == g].dropna(subset=[clave])
            if s.empty:
                continue
            pos = posiciones[g]
            bp = ax.boxplot(s[clave].values, positions=[pos], widths=0.6, patch_artist=True, showfliers=False,
                            manage_ticks=False, medianprops=dict(color="black", lw=2),
                            whiskerprops=dict(color="#555555"), capprops=dict(color="#555555"))
            bp["boxes"][0].set_facecolor(COLORES[g])
            bp["boxes"][0].set_alpha(0.35)
            bp["boxes"][0].set_edgecolor("#555555")
            if s["Estirado_en_frio"].mean() > 0.5:
                bp["boxes"][0].set_hatch("//")
            for fam, (mk, _, _) in FAMILIAS.items():
                sf = s[s["Familia_SAE"] == fam]
                if not sf.empty:
                    ax.scatter(pos + rng.uniform(-0.2, 0.2, len(sf)), sf[clave], marker=mk, c=COLORES[g], s=38,
                               alpha=0.9, edgecolors="white", linewidths=0.5, zorder=3)
            ax.scatter([pos], [s[clave].mean()], marker="D", s=75, c="white", edgecolors="black", linewidths=1.3, zorder=5)
            if resaltar:
                f = s[s["flag_anomalia"]]
                if not f.empty:
                    ax.scatter([pos] * len(f), f[clave], s=190, facecolors="none", edgecolors=ROJO, linewidths=1.8, zorder=6)
            if not atipicos.empty:
                for _, o in atipicos[(atipicos["Condicion_Grupo"] == g) & (atipicos["clave"] == clave)].iterrows():
                    ax.annotate(o["SAE Grade"], (pos, o["valor"]), xytext=(9, 4), textcoords="offset points", fontsize=8.5)
        ax.set_xticks([posiciones[g] for g in grupos])
        ax.set_xticklabels([f"{g.replace(' + ', chr(10) + '+ ')}\n(n = {int(d.loc[d['Condicion_Grupo'] == g, clave].notna().sum())})"
                            for g in grupos], fontsize=8.8)
        ax.set_ylabel(etiqueta)
        ax.set_title(f"({'abcd'[list(VARIABLES).index(clave)]}) {etiqueta}", loc="left", fontsize=11.5, pad=24)
        if "_gap" in posiciones:
            ax.axvline(posiciones["_gap"] - 0.6, color="#888888", ls=":", lw=1.2)
            puros = [posiciones[g] for g in grupos if g in GRUPOS_PUROS]
            comb = [posiciones[g] for g in grupos if g not in GRUPOS_PUROS]
            tr = ax.get_xaxis_transform()
            if puros:
                ax.text(np.mean(puros), 1.015, "Térmico puro", transform=tr, ha="center", va="bottom", fontsize=9.5,
                        fontweight="bold", color="#333333")
            if comb:
                ax.text(np.mean(comb), 1.015, "Térmico + estirado en frío", transform=tr, ha="center", va="bottom",
                        fontsize=9.5, fontweight="bold", color="#333333")
    h = [Line2D([], [], marker="D", ls="", markerfacecolor="white", markeredgecolor="black", label="Media", markersize=8),
         Line2D([], [], color="black", lw=2, label="Mediana")]
    usadas = set(d["Familia_SAE"])
    h += [Line2D([], [], marker=mk, ls="", color="#555555", label=f"Familia {lab}", markersize=8)
          for fam, (mk, lab, _) in FAMILIAS.items() if fam in usadas]
    if resaltar and d["flag_anomalia"].any():
        h.append(Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=ROJO, markeredgewidth=1.8,
                        markersize=12, label="Registro con flag_anomalia"))
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 0.075))
    fig.suptitle("Recocido y normalizado: propiedades mecánicas por grupo", fontsize=14, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.2, hspace=0.42, wspace=0.2)
    pie_de_figura(fig, "Normalizado (perlita fina, grano refinado) supera al recocido (perlita gruesa) en UTS, YS y dureza, con menor "
                       "elongación; los grupos combinados con estirado en frío (rayados) están endurecidos por deformación y son "
                       "10xx-15xx, por lo que no son comparables con los aleados térmicos puros. Etiquetas = grado SAE de valores "
                       "atípicos (regla 1.5×IQR, grupos con n ≥ 6). " + etiqueta_modo(modo, n_excl))
    return guardar_figura(fig, "fig03_02_boxplots_recocido_normalizado.png", dpi)


def fig_tendencias(d: pd.DataFrame, resaltar: bool, modo: str, n_excl: int, dpi: int) -> Path:
    """Figura 3: propiedad vs %C por grupo, con regresión donde es informativa."""
    grupos = grupos_presentes(d)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    for ax, (clave, etiqueta) in zip(axes.ravel(), VARIABLES.items()):
        validos = d["C_pct"].notna() & d[clave].notna()
        for g in grupos:
            s = d[validos & (d["Condicion_Grupo"] == g)]
            if s.empty:
                continue
            mk = "o" if g in GRUPOS_PUROS else "s"
            e = estadisticos_relacion(s["C_pct"], s[clave])
            rango = s["C_pct"].max() - s["C_pct"].min()
            etiq = g
            if e["n"] >= N_MIN_TENDENCIA and rango >= RANGO_C_TENDENCIA - 1e-9 and pd.notna(e["pendiente"]):
                xs = np.array([s["C_pct"].min(), s["C_pct"].max()])
                ax.plot(xs, e["intercepto"] + e["pendiente"] * xs, color=COLORES[g], lw=2.2, zorder=4)
                etiq = f"{g}: {e['pendiente_por_0_1C']:+.1f} {UNIDAD_CORTA[clave]}/0.1 %C (r = {e['pearson_r']:+.2f}, n = {e['n']})"
            else:
                etiq = f"{g} (n = {e['n']}; sin recta: rango de %C o n insuficiente)"
            ax.scatter(s["C_pct"], s[clave], c=COLORES[g], marker=mk, s=50, alpha=0.85, edgecolors="white",
                       linewidths=0.6, label=etiq, zorder=3)
            if resaltar:
                f = s[s["flag_anomalia"]]
                if not f.empty:
                    ax.scatter(f["C_pct"], f[clave], s=210, facecolors="none", edgecolors=ROJO, linewidths=1.8, zorder=5)
        ax.set_xlabel("Carbono, %C (promedio de límites Min-Max, % en masa)")
        ax.set_ylabel(etiqueta)
        ax.set_title(f"({'abcd'[list(VARIABLES).index(clave)]}) {etiqueta} vs %C", loc="left", fontsize=11.5)
        ax.legend(fontsize=7.6, loc="best", framealpha=0.92)
    fig.suptitle("Tendencia con el %C dentro de cada grupo de recocido y normalizado", fontsize=14, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.13, hspace=0.3, wspace=0.2)
    pie_de_figura(fig, "Más %C implica más perlita/cementita: sube la resistencia y baja la elongación dentro de cada tratamiento. La "
                       "recta solo se dibuja con n ≥ 5 y rango de %C ≥ 0.15; los grupos de %C casi constante no permiten separar el "
                       "efecto del carbono. Círculos = térmico puro, cuadrados = con estirado en frío. " + etiqueta_modo(modo, n_excl),
                  ancho=160)
    return guardar_figura(fig, "fig03_03_tendencia_vs_carbono.png", dpi)


def fig_pares(detalle: pd.DataFrame, resumen: pd.DataFrame, resaltar: bool, dpi: int) -> Path | None:
    """Figura 4: Annealed → Normalized para cada grado (mismo SAE Grade)."""
    if detalle.empty:
        return None
    det = detalle.sort_values(["C_pct", "SAE Grade"]).reset_index(drop=True)
    y = np.arange(len(det))
    fig, axes = plt.subplots(1, 4, figsize=(19, 0.42 * len(det) + 4.5), sharey=True)
    for ax, (clave, etiqueta) in zip(axes, VARIABLES.items()):
        a, b = det[f"{clave}_Annealed"], det[f"{clave}_Normalized"]
        for i in y:
            if pd.notna(a[i]) and pd.notna(b[i]):
                ax.plot([a[i], b[i]], [i, i], color="#999999", lw=1.4, zorder=2)
        ax.scatter(a, y, c=COLORES["Annealed"], s=52, zorder=3, edgecolors="white", linewidths=0.5)
        ax.scatter(b, y, c=COLORES["Normalized"], s=52, zorder=3, edgecolors="white", linewidths=0.5)
        if resaltar:
            for i in y[det["flag_anomalia"].to_numpy()]:
                ax.scatter([a[i], b[i]], [i, i], s=170, facecolors="none", edgecolors=ROJO, linewidths=1.6, zorder=6)
        fila = resumen[resumen["clave"] == clave]
        titulo = etiqueta
        if not fila.empty and "delta_medio" in fila and pd.notna(fila.iloc[0].get("delta_medio")):
            r = fila.iloc[0]
            titulo += f"\nΔ medio = {r['delta_medio']:+.1f} {UNIDAD_CORTA[clave]} (p {p_texto(r['p_holm'])}, {r['prueba_recomendada']})"
        ax.set_title(titulo, fontsize=10.2)
        ax.set_xlabel(etiqueta)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{g}  ({c:.2f} %C)" for g, c in zip(det["SAE Grade"], det["C_pct"])], fontsize=9)
    for lbl, flag in zip(axes[0].get_yticklabels(), det["flag_anomalia"]):
        if flag and resaltar:
            lbl.set_color(ROJO)
    axes[0].invert_yaxis()
    h = [Line2D([], [], marker="o", ls="", color=COLORES["Annealed"], label="Annealed", markersize=9),
         Line2D([], [], marker="o", ls="", color=COLORES["Normalized"], label="Normalized", markersize=9)]
    if resaltar and det["flag_anomalia"].any():
        h.append(Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=ROJO, markeredgewidth=1.6,
                        markersize=12, label="Grado con flag_anomalia"))
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 0.03))
    fig.suptitle("Recocido → Normalizado en el mismo grado SAE (composición nominal idéntica), ordenado por %C",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.1, right=0.99, top=0.9, bottom=0.12, wspace=0.08)
    pie_de_figura(fig, "Cada fila es un grado: la composición se mantiene y cambia el ciclo térmico (enfriamiento al aire y temperatura "
                       "algo mayor en normalizado). El salto es mayor en aceros de alta templabilidad (p. ej. 4340, 4150, 4140), donde "
                       "el enfriamiento al aire puede producir bainita además de perlita fina.", ancho=170, y=0.0)
    return guardar_figura(fig, "fig03_04_pares_recocido_normalizado_por_grado.png", dpi)


def fig_supuestos(d: pd.DataFrame, detalle: pd.DataFrame, dpi: int) -> Path:
    """Figura 5: gráficos Q-Q de los residuos del ANOVA (D1) y de las diferencias pareadas."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    def qq(ax, x, titulo):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if len(x) < 3 or np.ptp(x) == 0:
            ax.text(0.5, 0.5, "no calculable", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(titulo, fontsize=10)
            return
        (osm, osr), (pend, inter, _) = stats.probplot(x, dist="norm")
        ax.scatter(osm, osr, s=32, color="#0072B2", alpha=0.85, edgecolors="white", linewidths=0.4, zorder=3)
        ax.plot(osm, pend * osm + inter, color="black", ls="--", lw=1.5)
        p, _ = prueba_shapiro(x)
        ax.set_title(titulo, fontsize=10)
        ax.text(0.04, 0.95, f"Shapiro p = {p_texto(p)}\nn = {len(x)}", transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#999999", alpha=0.92))
        ax.set_xlabel("Cuantiles teóricos (normal)")

    for ax, (clave, etiqueta) in zip(axes[0], VARIABLES.items()):
        res = []
        for g in GRUPOS_PUROS:
            v = d.loc[d["Condicion_Grupo"] == g, clave].dropna().to_numpy(dtype=float)
            res.append(v - v.mean())
        qq(ax, np.concatenate(res) if res else [], f"{etiqueta}\nresiduos ANOVA (Annealed vs Normalized)")
        ax.set_ylabel("Residuos ordenados")
    for ax, (clave, etiqueta) in zip(axes[1], VARIABLES.items()):
        col = f"delta_{clave}"
        qq(ax, detalle[col] if (not detalle.empty and col in detalle) else [], f"{etiqueta}\ndiferencias pareadas (Normalized - Annealed)")
        ax.set_ylabel("Diferencias ordenadas")
    fig.suptitle("Verificación de supuestos de normalidad (D1: Annealed vs Normalized)", fontsize=14, fontweight="bold", y=1.0)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.2, hspace=0.45, wspace=0.28)
    pie_de_figura(fig, "Puntos cercanos a la recta indican normalidad aproximada. Fila superior: residuos del ANOVA independiente "
                       "(mezclan la variabilidad entre grados). Fila inferior: diferencias dentro de cada grado, que es lo que "
                       "supone la prueba t pareada. Con n = 20 Shapiro-Wilk detecta solo desviaciones moderadas o grandes.", ancho=165)
    return guardar_figura(fig, "fig03_05_supuestos_qq.png", dpi)


def fig_resumen_anova(tabla: pd.DataFrame, dpi: int) -> Path:
    """Figura 6: tamaño de efecto (η²) y p recomendado por diseño y variable, con el veredicto de validez."""
    disenos = [dz for dz in DISENOS if dz["id"] in set(tabla["diseno_id"])]
    cols = list(VARIABLES)
    eta = np.full((len(disenos), len(cols)), np.nan)
    anot = np.empty(eta.shape, dtype=object)
    for i, dz in enumerate(disenos):
        for j, c in enumerate(cols):
            f = tabla[(tabla["diseno_id"] == dz["id"]) & (tabla["clave"] == c)].iloc[0]
            if f["prueba_recomendada"] == "No evaluable":
                anot[i, j] = "no evaluable\n(n < 3)"
            else:
                eta[i, j] = f["eta2"]
                abrev = {"ANOVA clásico": "ANOVA", "ANOVA de Welch": "Welch", "Kruskal-Wallis": "K-W"}[f["prueba_recomendada"]]
                anot[i, j] = f"η² = {num(f['eta2'])}\np = {p_texto(f['p_holm'])} ({abrev})"
    fig, ax = plt.subplots(figsize=(11.5, 1.55 * len(disenos) + 3.6))
    sns.heatmap(pd.DataFrame(eta), annot=anot, fmt="", cmap="Blues", vmin=0, vmax=1, linewidths=1.2, linecolor="white",
                cbar_kws={"label": "η² (proporción de varianza explicada por el grupo)"}, ax=ax)
    for i in range(len(disenos)):
        for j in range(len(cols)):
            if np.isnan(eta[i, j]):
                ax.text(j + 0.5, i + 0.5, anot[i, j], ha="center", va="center", fontsize=9, color="#555555")
    etiquetas = []
    for i, dz in enumerate(disenos):
        nivel = tabla[tabla["diseno_id"] == dz["id"]].iloc[0]["nivel_validez"]
        corto = {"VÁLIDO CON RESERVAS": "válido con reservas", "NO EVALUABLE": "no evaluable"}.get(nivel, "NO VÁLIDO (confusión)")
        etiquetas.append(f"{dz['id']} · {dz['nombre']}\n[{corto}]")
        if not nivel.startswith("VÁLIDO"):
            ax.add_patch(Rectangle((0, i), len(cols), 1, fill=False, hatch="///", edgecolor="#999999", lw=0, zorder=5))
    ax.set_yticklabels(etiquetas, rotation=0, fontsize=9.5)
    ax.set_xticklabels([VARIABLES[c] for c in cols], rotation=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Resumen ANOVA: tamaño de efecto y validez del diseño", fontsize=13, fontweight="bold", pad=14)
    fig.subplots_adjust(left=0.36, right=0.97, top=0.9, bottom=0.2)
    pie_de_figura(fig, "p = valor p de la prueba recomendada tras verificar supuestos, ajustado por Holm entre las 4 variables. "
                       "Las filas rayadas son diseños donde los grupos difieren también en %C, familia SAE o estirado en frío: un η² "
                       "grande ahí NO es efecto del tratamiento térmico. En D1 la inferencia principal es la prueba pareada (fig03_04).",
                  ancho=135, y=0.0)
    return guardar_figura(fig, "fig03_06_resumen_anova_validez.png", dpi)


# ----------------------------------------------------------------------------
# 11. RESUMEN INTERPRETADO
# ----------------------------------------------------------------------------
def lineas_anova_variable(r: dict, alfa: float) -> list[str]:
    """Líneas de texto para una variable de un diseño."""
    if r["prueba_recomendada"] == "No evaluable":
        return [f"  {r['variable']:<22} no evaluable ({r['n_por_grupo']})"]
    norm = {True: "se cumple", False: "NO se cumple", None: "no verificable"}[r["normalidad_ok"]]
    var = {True: "se cumple", False: "NO se cumple", None: "no verificable"}[r["varianzas_ok"]]
    L = [f"  {r['variable']:<22} ANOVA F({int(r['anova_df1'])}, {int(r['anova_df2'])}) = {num(r['anova_F'])}, "
         f"p {p_texto(r['anova_p'])}, η² = {num(r['eta2'])} ({r['efecto_eta2']}), ω² = {num(r['omega2'])}"]
    sh = r["shapiro_p_por_grupo"] + (f"  [no verificable: {r['shapiro_no_verificable']}]" if r["shapiro_no_verificable"] else "")
    L.append(f"      Normalidad (Shapiro por grupo): {sh} -> {norm}")
    L.append(f"      Homogeneidad de varianzas (Levene): p {p_texto(r['levene_p'])} -> {var}")
    L.append(f"      Respaldo: Welch F = {num(r['welch_F'])} (p {p_texto(r['welch_p'])}); "
             f"Kruskal-Wallis H = {num(r['kw_H'])}, p {p_texto(r['kw_p'])}, ε² = {num(r['kw_eps2'])}")
    conc = {True: "ANOVA y Kruskal-Wallis coinciden", False: "ANOVA y Kruskal-Wallis DISCREPAN: interpretar con cautela",
            None: "concordancia no evaluable"}[r.get("concordancia_anova_kw")]
    L.append(f"      -> Prueba recomendada: {r['prueba_recomendada']}; p = {p_texto(r['p_recomendado'])}, "
             f"p Holm = {p_texto(r['p_holm'])} ({'significativo' if pd.notna(r['p_holm']) and r['p_holm'] < alfa else 'no significativo'} "
             f"a α = {alfa}); {conc}")
    if r["n_min"] < N_SHAPIRO_POTENCIA:
        L.append(f"      Reserva: n mínimo = {r['n_min']} (< {N_SHAPIRO_POTENCIA}): Shapiro y Levene tienen baja potencia.")
    return L


def construir_resumen(args, d_completo, d, foco_completo, foco, diag_grupos, est, tend, atipicos, anova_tabla, posthoc,
                      resultados_diseno, detalle_pares, resumen_pares, sens, rutas) -> str:
    L: list[str] = []
    sep = "=" * 96

    def titulo(t: str) -> None:
        L.extend(["", sep, t, sep])

    def parrafo(t: str, sangria: str = "  - ", cont: str = "    ", ancho: int = 94) -> None:
        L.append(textwrap.fill(t, ancho, initial_indent=sangria, subsequent_indent=cont))

    L.append(sep)
    L.append("RESUMEN 03 · RECOCIDO, NORMALIZADO Y ANOVA")
    L.append(sep)
    L.append(f"Valores de resistencia: {args.valores} | Anomalías: {args.anomalias} | α = {args.alfa} | n mínimo por grupo = {args.min_n}")
    L.append(f"Registros de recocido/normalizado: {len(foco_completo)} de {len(d_completo)} (en el análisis: {len(foco)}); "
             f"con flag_anomalia: {int(foco_completo['flag_anomalia'].sum())}")

    # --- 1. Diagnóstico ------------------------------------------------------------------
    titulo("1. DIAGNÓSTICO DE LIMITACIONES DE LOS DATOS")
    L.append("Grupos (registros, grados, tipo de proceso, %C, T de tratamiento):")
    for _, g in diag_grupos.iterrows():
        temp = f"{g['Temp_C_media']:.0f} °C" if pd.notna(g["Temp_C_media"]) else "NA"
        L.append(f"  {g['Condicion_Grupo']:<24} n={int(g['n_registros']):>2} | {g['Tipo_Proceso']:<27} | "
                 f"%C {g['C_pct_media']:.2f} ({g['C_pct_min']:.2f}-{g['C_pct_max']:.2f}) | aleados {g['pct_aleados']:.0f} % | "
                 f"T media {temp} | flags {int(g['n_flag_anomalia'])} | atípicos IQR {int(g['n_atipicos_IQR'])}")

    n_grupo = diag_grupos.set_index("Condicion_Grupo")["n_registros"]
    n_puro = int(diag_grupos[diag_grupos["Tipo_Proceso"] == "Térmico puro"]["n_registros"].sum())
    n_comb = int(diag_grupos[diag_grupos["Tipo_Proceso"] != "Térmico puro"]["n_registros"].sum())
    L.append("")
    parrafo(f"L1 · Tamaños de muestra desiguales: {', '.join(f'{g} = {int(n)}' for g, n in n_grupo.items())} "
            f"(razón máx./mín. = {n_grupo.max() / n_grupo.min():.1f}). Normalized + Cold drawn tiene n = {int(n_grupo.get('Normalized + Cold drawn', 0))}: "
            "no admite ninguna prueba estadística.", "  ", "  ")
    comb = diag_grupos[diag_grupos["Tipo_Proceso"] != "Térmico puro"]
    parrafo(f"L2 · Mezcla de tratamiento térmico puro con estirado en frío: {n_puro} registros son térmicos puros y {n_comb} "
            f"({100 * n_comb / (n_puro + n_comb):.0f} %) incluyen estirado en frío. Condiciones combinadas presentes: "
            + "; ".join(f"'{r['condicion_mas_frecuente']}'" for _, r in comb.iterrows())
            + ". El grupo 'Spheroidized annealed' parece solo térmico por su nombre, pero todos sus registros están estirados en frío.",
            "  ", "  ")
    al = diag_grupos.set_index("Condicion_Grupo")["pct_aleados"]
    parrafo("L3 · Familias SAE distintas: "
            + ", ".join(f"{g} {al[g]:.0f} % aleados" for g in al.index) +
            ". Los grupos térmicos puros son aceros aleados 3xxx-9xxx (más 1340), mientras los combinados son 10xx/15xx. "
            "Además el %C de Spheroidized annealed (0.60-0.97) casi no se solapa con el de los demás: no hay soporte común "
            "para separar efecto del tratamiento y efecto del carbono.", "  ", "  ")
    temps = diag_grupos.set_index("Condicion_Grupo")["Temp_C_media"]
    if pd.notna(temps.get("Annealed")) and pd.notna(temps.get("Normalized")):
        parrafo(f"L4 · Temperatura de tratamiento: la T media de normalizado ({temps['Normalized']:.0f} °C) supera a la de recocido "
                f"({temps['Annealed']:.0f} °C): la comparación recocido-normalizado mide el ciclo completo (temperatura + velocidad de "
                "enfriamiento), no solo la velocidad de enfriamiento.", "  ", "  ")
    n_flag = int(foco_completo["flag_anomalia"].sum())
    flag_txt = ", ".join(f"#{int(r['registro'])} SAE {r['SAE Grade']} ({r['Condicion_Grupo']})"
                         for _, r in foco_completo[foco_completo["flag_anomalia"]].iterrows())
    parrafo(f"L5 · Anomalías y atípicos: {n_flag} registros del foco tienen flag_anomalia ({flag_txt}). "
            f"Atípicos por 1.5×IQR (grupos con n ≥ {N_MIN_ATIPICOS}): {len(atipicos)}"
            + (" (" + ", ".join(f"{r['SAE Grade']} {r['variable'].split(' (')[0]} en {r['Condicion_Grupo']}" for _, r in atipicos.iterrows()) + ")"
               if len(atipicos) else "") + ". En varios casos la anomalía afecta solo a una columna (p. ej. ksi o HRB) y no a las "
            "variables analizadas; ver sensibilidad en la sección 5.", "  ", "  ")
    const = diag_grupos[diag_grupos["variables_sin_variabilidad"] != ""]
    if not const.empty:
        parrafo("L6 · Valores sin variabilidad (tabulados): "
                + "; ".join(f"{r['Condicion_Grupo']}: {r['variables_sin_variabilidad']}" for _, r in const.iterrows())
                + ". Un valor idéntico en todos los grados sugiere un dato tabulado/redondeado, no una medición, y vuelve "
                "incalculables Shapiro y Levene en esos grupos.", "  ", "  ")
    parrafo("L7 · Los datos son valores nominales de tablas SAE (un renglón por grado, sin réplicas): no hay variabilidad de "
            "ensayo y la independencia es solo aproximada.", "  ", "  ")

    # --- 2. Exploración --------------------------------------------------------------------
    titulo("2. EXPLORACIÓN DESCRIPTIVA (media ± desv. est.)")
    for g in grupos_presentes(foco):
        partes = []
        for etiqueta, corto in (("UTS (MPa)", "UTS"), ("YS (MPa)", "YS"), ("Dureza Brinell (HB)", "HB"), ("Elongación (%)", "Elong")):
            f = est[(est["Condicion_Grupo"] == g) & (est["variable"] == etiqueta)].iloc[0]
            if f["n"] == 0:
                partes.append(f"{corto} NA")
                continue
            dec = 1 if corto == "Elong" else 0
            sd = "NA" if pd.isna(f["desv_est"]) else f"{f['desv_est']:.{dec}f}"
            partes.append(f"{corto} {f['media']:.{dec}f}±{sd}")
        n_g = int(est[(est["Condicion_Grupo"] == g)]["n"].max())
        L.append(f"  {g:<24} (n={n_g:>2}) " + " | ".join(partes))

    L.append("\nTendencia con el %C dentro de cada grupo (pendiente por +0.10 %C; † = poco informativa):")
    for g in grupos_presentes(foco):
        t = tend[tend["Condicion_Grupo"] == g]
        partes = []
        for _, r in t.iterrows():
            corto = r["variable"].split(" (")[0].replace("Dureza Brinell", "HB").replace("Elongación", "Elong")
            if pd.isna(r["pearson_r"]):
                partes.append(f"{corto} NA†")
            else:
                partes.append(f"{corto} {r['pendiente_por_0_1C']:+.1f} (r={r['pearson_r']:+.2f}){'†' if r['nota'] else ''}")
        L.append(f"  {g:<24} " + " | ".join(partes))
    parrafo("Lectura: dentro de cada grupo, más %C implica más perlita/cementita (o carburos), por lo que suben UTS, YS y dureza y "
            "baja la elongación, siempre que el rango de %C del grupo sea amplio. Los grupos † no permiten separar el efecto del carbono.",
            "  ", "  ")

    # --- 3. ANOVA -------------------------------------------------------------------------------
    titulo("3. ANOVA, SUPUESTOS Y PRUEBAS DE RESPALDO POR DISEÑO")
    for dz in DISENOS:
        regs, nivel, razones, diag = resultados_diseno[dz["id"]]
        L.append(f"\n{dz['id']} · {dz['nombre']}")
        L.append("  Grupos: " + " | ".join(f"{g} (n={diag['n'][g]})" for g in dz["grupos"]))
        L.append(f"  VEREDICTO DE VALIDEZ: {nivel}")
        for r in razones:
            parrafo(r, "     · ", "       ", 90)
        if nivel == "NO EVALUABLE":
            continue
        if nivel.startswith("NO VÁLIDO"):
            parrafo("Los resultados siguientes son DESCRIPTIVOS: la diferencia entre grupos no puede atribuirse al tratamiento térmico.",
                    "  ! ", "    ", 92)
        for c in VARIABLES:
            r = next(x for x in regs if x["clave"] == c)
            L.extend(lineas_anova_variable(r, args.alfa))
        ph = posthoc[(posthoc["diseno_id"] == dz["id"]) & (posthoc["recomendado"])] if not posthoc.empty else pd.DataFrame()
        if not ph.empty:
            L.append("  Comparaciones por pares (método según supuestos; descriptivas si el diseño es NO VÁLIDO):")
            for _, f in ph.iterrows():
                L.append(f"    {f['variable']:<22} {f['grupo_a']} vs {f['grupo_b']}: diferencia = {f['diferencia']:+.1f} "
                         f"({f['metodo']}, p ajustado {p_texto(f['p_ajustado'])})")

    # --- 4. Pareado ------------------------------------------------------------------------------
    titulo("4. ANÁLISIS PAREADO: Annealed vs Normalized en el mismo grado SAE")
    if resumen_pares.empty or "delta_medio" not in resumen_pares:
        L.append("  No hay pares suficientes.")
    else:
        L.append(f"  Pares disponibles: {int(resumen_pares['n_pares'].max())} grados (Normalized - Annealed). "
                 "El emparejamiento elimina la variabilidad entre grados (composición idéntica).")
        for _, r in resumen_pares.iterrows():
            L.append(f"  {r['variable']:<22} Δ medio = {r['delta_medio']:+7.1f} (IC95 % {r['ic95_inf']:+.1f} a {r['ic95_sup']:+.1f}; "
                     f"{r['delta_relativo_medio_pct']:+.0f} %) | dz = {num(r['d_cohen_dz'])} | suben {int(r['pares_con_aumento'])}/"
                     f"{int(r['n_pares'])} | Shapiro(Δ) p {p_texto(r['shapiro_p_diferencias'])} -> {r['prueba_recomendada']}, "
                     f"p {p_texto(r['p_recomendado'])}, p Holm {p_texto(r['p_holm'])}")
        if not detalle_pares.empty:
            top = detalle_pares.sort_values("delta_UTS", ascending=False).head(3)
            parrafo("Mayores saltos de UTS al normalizar: " + ", ".join(f"{r['SAE Grade']} ({r['delta_UTS']:+.0f} MPa)" for _, r in top.iterrows())
                    + ". Son aceros de alta templabilidad: el enfriamiento al aire puede formar bainita (y algo de martensita), "
                    "por lo que el efecto no es solo 'perlita más fina' y depende del grado.", "  ", "  ")
            dt = detalle_pares["delta_Temp_C"].mean()
            if pd.notna(dt):
                parrafo(f"La T de normalizado excede a la de recocido en promedio {dt:+.0f} °C en estos pares: el efecto medido es del ciclo "
                        "completo (temperatura de austenización + velocidad de enfriamiento).", "  ", "  ")
            rho_u = resumen_pares[resumen_pares["clave"] == "UTS"]
            if not rho_u.empty and pd.notna(rho_u.iloc[0].get("rho_delta_vs_C")):
                parrafo(f"Relación entre el salto de UTS y el %C: ρ de Spearman = {rho_u.iloc[0]['rho_delta_vs_C']:+.2f} "
                        f"(p {p_texto(rho_u.iloc[0]['p_rho_delta_vs_C'])}); un %C mayor tiende a acompañarse de mayor templabilidad "
                        "y de un mayor efecto del normalizado.", "  ", "  ")

    titulo("5. SENSIBILIDAD A ANOMALÍAS (D1: todos los registros vs sin los marcados)")
    for _, r in sens.iterrows():
        par = (f" | pareado: n={int(r['n_pares'])}, Δ={r['delta_medio_pareado']:+.1f}, p {p_texto(r['p_pareado'])}"
               if "n_pares" in r and pd.notna(r.get("n_pares")) else "")
        L.append(f"  {r['conjunto']:<14} {r['variable']:<22} N={int(r['n_registros']):>2} F={num(r['anova_F'])} "
                 f"p {p_texto(r['anova_p'])} η²={num(r['eta2'])}{par}")

    # --- 6. Interpretación -----------------------------------------------------------------------
    titulo("6. INTERPRETACIÓN METALÚRGICA")
    for t in TEXTO_METALURGICO:
        parrafo(t)

    titulo("7. CUÁNDO ES VÁLIDO EL ANOVA Y QUÉ RESERVAS TENER")
    for t in GUIA_ANOVA:
        parrafo(t)
    L.append("")
    L.append("Aplicación a este dataset:")
    v1 = resultados_diseno["D1"][1]
    parrafo(f"D1 ({v1}): es la única comparación sin confusión. Como los 20 grados aparecen en ambos tratamientos, "
            "la prueba pareada es la inferencia apropiada y el ANOVA independiente es una aproximación conservadora (pierde potencia "
            "porque la variabilidad entre grados es enorme frente al cambio dentro de cada grado).")
    parrafo(f"D2 ({resultados_diseno['D2'][1]}): el ANOVA se calcula, pero Spheroidized annealed difiere en %C, familia SAE y estirado en frío; "
            "su η² no es efecto del esferoidizado.")
    parrafo(f"D3 ({resultados_diseno['D3'][1]}): añadir estirado en frío coincide con otro %C y otra familia; no se puede aislar.")
    parrafo(f"D4 ({resultados_diseno['D4'][1]}): n = 2 en Normalized + Cold drawn.")
    parrafo("Para aislar el efecto de la deformación en frío a composición idéntica se usó el análisis por pares Hot rolled vs Cold drawn del "
            "script 02 (50 pares del mismo grado).")

    titulo("8. ARCHIVOS GENERADOS")
    for r in rutas:
        L.append(f"  - {r.relative_to(RAIZ)}")
    return "\n".join(L)


# ----------------------------------------------------------------------------
# 12. PROGRAMA PRINCIPAL
# ----------------------------------------------------------------------------
def guardar_tabla(df: pd.DataFrame, nombre: str) -> Path:
    DIR_TABLAS.mkdir(parents=True, exist_ok=True)
    ruta = DIR_TABLAS / nombre
    df.to_csv(ruta, index=False, encoding="utf-8-sig")
    return ruta


def main() -> None:
    args = leer_argumentos()
    configurar_estilo()

    d_completo = preparar_variables(cargar_datos(), usar_corregidos=(args.valores == "corregidos"))
    foco_completo = d_completo[d_completo["Condicion_Grupo"].isin(GRUPOS_FOCO)].copy()
    if foco_completo.empty:
        raise SystemExit("No hay registros de recocido ni normalizado en aceros_limpio.csv.")

    foco = foco_completo[~foco_completo["flag_anomalia"]].copy() if args.anomalias == "excluir" else foco_completo.copy()
    n_excl = len(foco_completo) - len(foco)
    resaltar = args.anomalias == "resaltar"

    # --- Diagnóstico y descriptivos -----------------------------------------------------------
    atipicos = detectar_atipicos(foco, grupos_presentes(foco))
    diag_grupos = tabla_diagnostico_grupos(foco, atipicos)
    est = estadisticas_grupos(foco)
    tend = tendencias_c(foco)

    # --- ANOVA por diseño ---------------------------------------------------------------------------
    regs_todos, posthoc_todos, resultados = [], [], {}
    for dz in DISENOS:
        regs, ph, diag, nivel, razones = probar_diseno(foco, dz, args.alfa, args.min_n)
        resultados[dz["id"]] = (regs, nivel, razones, diag)
        regs_todos.extend(regs)
        posthoc_todos.extend(ph)
    anova_tabla = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in regs_todos])
    posthoc = pd.DataFrame(posthoc_todos)

    # --- Pareado y sensibilidad ----------------------------------------------------------------------
    detalle_pares, resumen_pares = analisis_pareado(foco, "Annealed", "Normalized", args.alfa)
    sens = tabla_sensibilidad(foco_completo, args.alfa, args.min_n)

    tablas = [
        guardar_tabla(diag_grupos, "tabla_03_diagnostico_grupos.csv"),
        guardar_tabla(est, "tabla_03_estadisticas_grupos.csv"),
        guardar_tabla(tend, "tabla_03_tendencias_vs_carbono.csv"),
        guardar_tabla(atipicos, "tabla_03_atipicos.csv"),
        guardar_tabla(anova_tabla, "tabla_03_anova_supuestos.csv"),
        guardar_tabla(detalle_pares, "tabla_03_pares_recocido_normalizado.csv"),
        guardar_tabla(resumen_pares, "tabla_03_resumen_pares_recocido_normalizado.csv"),
        guardar_tabla(sens, "tabla_03_sensibilidad_anomalias.csv"),
    ]
    if not posthoc.empty:
        tablas.append(guardar_tabla(posthoc, "tabla_03_posthoc.csv"))

    # --- Figuras ----------------------------------------------------------------------------------------
    figuras = [
        fig_diagnostico(foco, diag_grupos, args.dpi),
        fig_boxplots(foco, atipicos, resaltar, args.anomalias, n_excl, args.dpi),
        fig_tendencias(foco, resaltar, args.anomalias, n_excl, args.dpi),
    ]
    f4 = fig_pares(detalle_pares, resumen_pares, resaltar, args.dpi)
    if f4 is not None:
        figuras.append(f4)
    figuras.append(fig_supuestos(foco, detalle_pares, args.dpi))
    figuras.append(fig_resumen_anova(anova_tabla, args.dpi))

    # --- Resumen ------------------------------------------------------------------------------------------
    DIR_OUTPUTS.mkdir(parents=True, exist_ok=True)
    ruta_resumen = DIR_OUTPUTS / "resumen_03_recocido_normalizado_anova.txt"
    resumen = construir_resumen(args, d_completo, foco, foco_completo, foco, diag_grupos, est, tend, atipicos,
                                anova_tabla, posthoc, resultados, detalle_pares, resumen_pares, sens,
                                tablas + figuras + [ruta_resumen])
    ruta_resumen.write_text(resumen + "\n", encoding="utf-8")
    print(resumen)


if __name__ == "__main__":
    main()