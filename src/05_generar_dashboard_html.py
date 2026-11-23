#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_generar_dashboard_html.py
============================
Genera un dashboard HTML interactivo y portátil para la base SAE de aceros.

Requisitos:
    pip install pandas numpy
    Los gráficos usan Plotly.js desde una CDN HTTPS (requiere internet al abrir el HTML).

Entrada obligatoria:
    data/processed/aceros_limpio.csv                (script 01)

Entradas opcionales:
    data/processed/auditoria_inconsistencias.csv    (script 01)
    data/processed/tabla_02_*.csv                   (script 02)
    outputs/tablas/tabla_03_*.csv                   (script 03)
    outputs/tablas/tabla_04_*.csv                   (script 04)
    outputs/resumen_02_carbono_procesos.txt         (script 02)
    outputs/resumen_03_recocido_normalizado_anova.txt (script 03)
    outputs/resumen_04_carbono_vs_aleados.txt       (script 04)

Salida:
    outputs/index.html (archivo único con datos, estilos y lógica incrustados)

Publicación:
    python src/05_generar_dashboard_html.py
    python src/05_generar_dashboard_html.py --copiar docs
    python src/05_generar_dashboard_html.py --copiar raiz
    python src/05_generar_dashboard_html.py --copiar ambos
    python src/05_generar_dashboard_html.py --anomalias excluir --valores originales

En GitHub Pages: configurar Settings > Pages > Deploy from a branch, y seleccionar
la rama principal y /docs (con --copiar docs) o /(root) (con --copiar raiz).
No se requiere Flask, servidor de Python, ni archivos JSON externos; los datos se
incrustan en el HTML. Los gráficos requieren acceso a la CDN de Plotly.js.

Convenciones heredadas de los scripts 01-04:
    - Los registros no se eliminan del archivo publicado.
    - El selector de anomalías (resaltar/excluir/incluir) es interactivo.
    - UTS y YS usan columnas corregidas por defecto, sin modificar las originales.
    - Hardness_HB_num contiene únicamente durezas Brinell comparables.
    - C_pct es una aproximación desde límites de composición (no valor de colada).
    - Los análisis precalculados de 02/03/04 NO cambian con filtros del navegador.
      Los gráficos y estadísticas que SÍ cambian están explícitamente señalados.

Seguridad y portabilidad:
    - Se escapa JSON incrustado para que ningún campo termine una etiqueta <script>.
    - La tabla exploratoria escapa los valores de texto antes de usar innerHTML.
    - No se solicitan recursos locales con fetch() (funciona con file://).
    - El HTML copiado a /docs o /(root) es idéntico, sin rutas locales absolutas.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# 1. DIRECTORIOS Y CONFIGURACIÓN
# -----------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[1]
DIR_PROCESSED = RAIZ / "data" / "processed"
DIR_OUTPUTS = RAIZ / "outputs"
DIR_TABLAS = DIR_OUTPUTS / "tablas"
ARCHIVO_DATOS = DIR_PROCESSED / "aceros_limpio.csv"
ARCHIVO_AUDITORIA = DIR_PROCESSED / "auditoria_inconsistencias.csv"
HTML_SALIDA = DIR_OUTPUTS / "index.html"

COLUMNAS_OBLIGATORIAS = [
    "registro", "SAE Grade", "Familia_SAE", "Condicion_Grupo", "C_pct",
    "UTS (MPa)", "YS (MPa)", "Hardness_HB_num", "Elongation (%)",
    "flag_anomalia",
]
COLUMNAS_NUMERICAS = [
    "registro", "C_pct", "UTS (MPa)", "YS (MPa)", "UTS_MPa_corr",
    "YS_MPa_corr", "Hardness_HB_num", "Hardness_raw_num", "Elongation (%)",
    "Reduction (%)", "Temp_C", "Temp_revenido_C", "n_anomalias",
] + [f"{e} ({lim})" for e in ("C", "Mn", "P", "S", "Si", "Ni", "Cr", "Mo", "Ti")
     for lim in ("Min", "Max")]

COLUMNAS_EXPORTAR = [
    "registro", "SAE Grade", "Familia_SAE", "Serie_SAE", "Condicion_Grupo",
    "Conditions_limpia", "Conditions_original", "C_pct", "C_pct_fuente",
    "UTS (MPa)", "YS (MPa)", "UTS_MPa_corr", "YS_MPa_corr",
    "Hardness_HB_num", "Hardness_raw_num", "Hardness_escala",
    "Elongation (%)", "Reduction (%)", "Temp_C", "Temp_revenido_C",
    "flag_anomalia", "n_anomalias", "motivo_auditoria",
] + [f"{el} ({lim})" for el in ("Cr", "Ni", "Mo", "Mn", "Si")
     for lim in ("Min", "Max")]

# Se leen las tablas que existen. Las ausentes no impiden generar el sitio.
# Nótese que las tablas 02 residen en data/processed, no en outputs/tablas.
TABLAS_OPCIONALES = {
    "02_correlaciones": (
        "02 · Correlaciones %C y propiedades",
        DIR_PROCESSED / "tabla_02_correlaciones_c_vs_propiedades.csv", "carbono"),
    "02_condiciones": (
        "02 · Estadísticas por condición",
        DIR_PROCESSED / "tabla_02_estadisticas_por_condicion.csv", "procesos"),
    "02_pares_frio": (
        "02 · Pares laminado vs. estirado",
        DIR_PROCESSED / "tabla_02_resumen_pares_hot_rolled_vs_cold_drawn.csv", "procesos"),
    "02_sensibilidad": (
        "02 · Sensibilidad a anomalías",
        DIR_PROCESSED / "tabla_02_sensibilidad_anomalias.csv", "auditoria"),
    "03_diagnostico": (
        "03 · Diagnóstico de grupos",
        DIR_TABLAS / "tabla_03_diagnostico_grupos.csv", "anova"),
    "03_estadisticas": (
        "03 · Estadísticas de tratamientos",
        DIR_TABLAS / "tabla_03_estadisticas_grupos.csv", "anova"),
    "03_anova": (
        "03 · ANOVA, supuestos y validez",
        DIR_TABLAS / "tabla_03_anova_supuestos.csv", "anova"),
    "03_pareados": (
        "03 · Resumen de pares recocido-normalizado",
        DIR_TABLAS / "tabla_03_resumen_pares_recocido_normalizado.csv", "anova"),
    "03_posthoc": (
        "03 · Comparaciones post hoc",
        DIR_TABLAS / "tabla_03_posthoc.csv", "anova"),
    "03_sensibilidad": (
        "03 · Sensibilidad a anomalías",
        DIR_TABLAS / "tabla_03_sensibilidad_anomalias.csv", "auditoria"),
    "04_descriptivos": (
        "04 · Descriptivos por familia",
        DIR_TABLAS / "tabla_04_descriptivos_por_familia.csv", "aleados"),
    "04_contraste": (
        "04 · Contraste global 10xx vs. aleados",
        DIR_TABLAS / "tabla_04_contraste_global_10xx_vs_aleados.csv", "aleados"),
    "04_comparables": (
        "04 · Condiciones comparables",
        DIR_TABLAS / "tabla_04_contrastes_condiciones_comparables.csv", "aleados"),
    "04_correlaciones": (
        "04 · Correlaciones por familia",
        DIR_TABLAS / "tabla_04_correlaciones_c_vs_propiedades.csv", "aleados"),
    "04_composicion": (
        "04 · Composición nominal de aleantes",
        DIR_TABLAS / "tabla_04_composicion_aleantes.csv", "aleados"),
    "04_diagnostico": (
        "04 · Diagnóstico de condiciones",
        DIR_TABLAS / "tabla_04_diagnostico_condiciones.csv", "aleados"),
    "04_sensibilidad": (
        "04 · Sensibilidad a anomalías",
        DIR_TABLAS / "tabla_04_sensibilidad_anomalias.csv", "auditoria"),
    "04_auditoria": (
        "04 · Hallazgos por familia",
        DIR_TABLAS / "tabla_04_auditoria_por_familia.csv", "auditoria"),
    "01_auditoria": (
        "01 · Auditoría original por registro",
        ARCHIVO_AUDITORIA, "auditoria"),
}

RESUMENES = {
    "carbono": DIR_OUTPUTS / "resumen_02_carbono_procesos.txt",
    "anova": DIR_OUTPUTS / "resumen_03_recocido_normalizado_anova.txt",
    "aleados": DIR_OUTPUTS / "resumen_04_carbono_vs_aleados.txt",
}


def argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Genera el dashboard SAE interactivo en outputs/index.html para GitHub Pages."
    )
    p.add_argument(
        "--anomalias", choices=("resaltar", "excluir", "incluir"), default="resaltar",
        help="Modo inicial del filtro del navegador; permite cambiarlo después."
    )
    p.add_argument(
        "--valores", choices=("corregidos", "originales"), default="corregidos",
        help="Modo inicial de UTS/YS; permite cambiarlo después."
    )
    p.add_argument(
        "--copiar", choices=("ninguno", "docs", "raiz", "ambos"), default="ninguno",
        help="Además de outputs/index.html, copia el mismo HTML a docs/ y/o al raíz."
    )
    return p.parse_args()


# -----------------------------------------------------------------------------
# 2. LECTURA, VALIDACIÓN, INTEGRACIÓN DE AUDITORÍA
# -----------------------------------------------------------------------------
def leer_csv(ruta: Path) -> pd.DataFrame:
    """Compatibilidad con archivos generados en utf-8-sig desde Windows/Excel."""
    try:
        return pd.read_csv(ruta, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(ruta, encoding="latin-1", low_memory=False)


def bool_serie(serie: pd.Series) -> pd.Series:
    return serie.astype("string").str.strip().str.lower().isin(("true", "1", "sí", "si"))


def cargar_datos() -> tuple[pd.DataFrame, bool]:
    if not ARCHIVO_DATOS.exists():
        raise FileNotFoundError(
            f"No existe {ARCHIVO_DATOS}. Ejecuta primero: python src/01_limpieza_datos.py"
        )
    df = leer_csv(ARCHIVO_DATOS)
    faltantes = [col for col in COLUMNAS_OBLIGATORIAS if col not in df.columns]
    if faltantes:
        raise ValueError(
            "Faltan columnas obligatorias en aceros_limpio.csv: " + ", ".join(faltantes)
        )
    if df.empty:
        raise ValueError("El archivo aceros_limpio.csv está vacío.")

    df = df.copy()
    df["SAE Grade"] = df["SAE Grade"].astype("string").fillna("")
    for col in ("Familia_SAE", "Condicion_Grupo", "motivo_auditoria"):
        if col not in df.columns:
            df[col] = "" if col == "motivo_auditoria" else "Sin clasificar"
        df[col] = df[col].astype("string").fillna("")
    df["flag_anomalia"] = bool_serie(df["flag_anomalia"])

    for col in COLUMNAS_NUMERICAS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # No se supone que toda instalación conserve exactamente los mismos scripts;
    # se mantienen ambas resistencias aunque no estén las columnas corregidas.
    for corregida, original in (
        ("UTS_MPa_corr", "UTS (MPa)"), ("YS_MPa_corr", "YS (MPa)")
    ):
        if corregida not in df.columns:
            print(f"[AVISO] Sin {corregida}; se utilizará {original} en su lugar.")
            df[corregida] = df[original]
        else:
            df[corregida] = df[corregida].fillna(df[original])

    auditoria_disponible = ARCHIVO_AUDITORIA.exists()
    if auditoria_disponible:
        audit = leer_csv(ARCHIVO_AUDITORIA)
        # El archivo limpio ya trae hallazgos, pero los completamos si faltan.
        if {"registro", "motivo_auditoria"}.issubset(audit.columns):
            motivos = (audit.drop_duplicates("registro")
                       .set_index("registro")["motivo_auditoria"].fillna(""))
            faltan_motivos = df["motivo_auditoria"].str.strip().eq("")
            df.loc[faltan_motivos, "motivo_auditoria"] = (
                df.loc[faltan_motivos, "registro"].map(motivos).fillna("").astype("string")
            )

    if df["registro"].duplicated().any():
        raise ValueError("La columna 'registro' no puede tener valores duplicados.")
    if df["C_pct"].notna().sum() == 0:
        raise ValueError("No hay valores numéricos de C_pct para graficar.")

    return df, auditoria_disponible


def valor_json(x: Any) -> Any:
    """Sanea tipos NumPy/pandas y números no finitos para JSON estándar."""
    if x is None or x is pd.NA:
        return None
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.isoformat()
    if isinstance(x, (list, tuple)):
        return [valor_json(v) for v in x]
    if isinstance(x, dict):
        return {str(k): valor_json(v) for k, v in x.items()}
    if pd.isna(x):
        return None
    return str(x)


def filas_json(df: pd.DataFrame) -> list[dict]:
    """Evita que NaN/Infinity pasen al JavaScript como tokens inválidos."""
    limpio = df.astype(object).where(pd.notna(df), None)
    return [
        {str(clave): valor_json(val) for clave, val in fila.items()}
        for fila in limpio.to_dict(orient="records")
    ]


def cargar_referencias() -> tuple[dict, dict, list[str]]:
    tablas: dict[str, dict] = {}
    omitidas: list[str] = []
    for clave, (nombre, ruta, seccion) in TABLAS_OPCIONALES.items():
        if not ruta.exists():
            omitidas.append(str(ruta.relative_to(RAIZ)))
            continue
        try:
            df = leer_csv(ruta)
        except (pd.errors.EmptyDataError, OSError, ValueError) as exc:
            print(f"[AVISO] No se pudo leer {ruta.name}: {exc}")
            omitidas.append(str(ruta.relative_to(RAIZ)))
            continue
        tablas[clave] = {
            "nombre": nombre,
            "seccion": seccion,
            "ruta": str(ruta.relative_to(RAIZ)).replace("\\", "/"),
            "columnas": [str(c) for c in df.columns],
            "filas": filas_json(df),
        }

    resumenes: dict[str, str] = {}
    for clave, ruta in RESUMENES.items():
        if ruta.exists():
            try:
                resumenes[clave] = ruta.read_text(encoding="utf-8-sig")
            except UnicodeDecodeError:
                resumenes[clave] = ruta.read_text(encoding="latin-1")
        else:
            omitidas.append(str(ruta.relative_to(RAIZ)))
    return tablas, resumenes, omitidas


def preparar_paquete(df: pd.DataFrame, auditoria: bool, args: argparse.Namespace) -> tuple[dict, list[str]]:
    tablas, resumenes, omitidas = cargar_referencias()
    columnas = [c for c in COLUMNAS_EXPORTAR if c in df.columns]
    datos = df[columnas].copy()
    carbono = df["C_pct"].dropna().astype(float)
    minimo, maximo = float(carbono.min()), float(carbono.max())
    # Evita un intervalo de anchura cero en el filtro de carbono.
    if minimo == maximo:
        minimo -= 0.01
        maximo += 0.01

    paquete = {
        "meta": {
            "titulo": "Atlas de aceros SAE",
            "fecha_generacion": datetime.now().astimezone().isoformat(timespec="seconds"),
            "total": len(df),
            "anomalias": int(df["flag_anomalia"].sum()),
            "filas_con_hallazgos": int(df["motivo_auditoria"].str.strip().ne("").sum()),
            "auditoria_externa": auditoria,
            "carbono_min": minimo,
            "carbono_max": maximo,
            "modo_anomalias": args.anomalias,
            "modo_valores": args.valores,
            "tablas_disponibles": len(tablas),
        },
        "registros": filas_json(datos),
        "tablas": tablas,
        "resumenes": resumenes,
    }
    return paquete, omitidas


def escapar_json_script(paquete: dict) -> str:
    """Protección contra '</script>' o entidades HTML presentes en datos."""
    texto = json.dumps(paquete, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return (texto.replace("&", "\\u0026").replace("<", "\\u003c")
            .replace(">", "\\u003e").replace("\u2028", "\\u2028")
            .replace("\u2029", "\\u2029"))


# -----------------------------------------------------------------------------
# 3. PLANTILLA HTML / CSS / JAVASCRIPT (TODO EMBEBIDO)
# -----------------------------------------------------------------------------
PLANTILLA_HTML = r'''<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<meta name="description" content="Análisis interactivo de composición, propiedades mecánicas, procesos y auditoría de aceros SAE.">
<title>Atlas de aceros SAE | Análisis metalúrgico</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" defer></script>
<style>
:root {
  --fondo:#f4f7fb; --papel:#fff; --tinta:#14233b; --sub:#536780;
  --borde:#dce5ef; --azul:#135ab3; --azul-claro:#e9f1fd;
  --oscuro:#0a203f; --alerta:#b72e3a; --alerta-fondo:#fff4f3;
  --verde:#0e7864; --sombra:0 10px 28px rgba(23,46,79,.06);
}
* { box-sizing:border-box; }
html { scroll-behavior:smooth; }
body { margin:0; font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;
  background:var(--fondo); color:var(--tinta); }
button,select,input { font:inherit; }
button { cursor:pointer; }
a { color:#1256ab; }
.skip { position:absolute; top:-60px; left:15px; z-index:30; background:#fff; padding:8px 12px; }
.skip:focus { top:10px; }
.hero { background:linear-gradient(115deg,#091c36 0%,#103b69 65%,#176b89 100%);
  color:#fff; padding:34px max(22px,calc((100vw - 1280px)/2)) 28px; }
.hero .eyebrow { text-transform:uppercase; letter-spacing:.15em; font-size:11px;
  font-weight:750; color:#a9dbf2; }
h1 { font-size:clamp(27px,3.4vw,44px); line-height:1.15; letter-spacing:-.035em;
  margin:7px 0 12px; }
.hero p { max-width:760px; color:#deebf7; margin:0; font-size:15px; }
.hero-meta { display:flex; gap:10px; flex-wrap:wrap; margin-top:20px; }
.hero-meta span { font-size:12px; color:#d5ebfa; border:1px solid #85abc65a;
  background:#ffffff13; padding:5px 11px; border-radius:40px; }
.nav-outer { position:sticky; top:0; z-index:15; background:#fff;
  box-shadow:0 2px 10px #223e5e14; border-bottom:1px solid var(--borde); }
.nav { max-width:1320px; padding:0 18px; margin:auto; display:flex; overflow-x:auto; scrollbar-width:thin; }
.tab { flex-shrink:0; border:0; border-bottom:3px solid transparent; border-radius:0;
  background:none; padding:16px 18px 13px; color:#40546c; font-size:13.5px; font-weight:680; }
.tab:hover { background:#f0f5fc; color:var(--azul); }
.tab[aria-selected="true"] { color:#1159ae; border-bottom-color:#2074da; background:#f7fbff; }
.shell { max-width:1320px; margin:24px auto 70px; padding:0 18px; }
.filters,.panel,.note,.kpi { background:var(--papel); border:1px solid var(--borde);
  border-radius:15px; box-shadow:var(--sombra); }
.filters { padding:20px; margin-bottom:18px; }
.filter-heading { display:flex; justify-content:space-between; gap:12px; align-items:center;
  margin-bottom:14px; }
.filter-heading h2 { margin:0; font-size:16px; }
.filter-heading .helper { font-size:12px; color:var(--sub); }
.filter-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px 18px; }
.control { display:flex; flex-direction:column; gap:5px; min-width:0; }
.control label { font-size:12px; color:#465b72; font-weight:720; }
.control select,.control input { width:100%; min-height:41px; border:1px solid #cbd9e6;
  border-radius:9px; padding:8px 10px; background:#fff; color:#192c43; outline-offset:2px; }
.control input[type="number"] { font-variant-numeric:tabular-nums; }
.range-pair { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
.filter-bottom { display:flex; flex-wrap:wrap; gap:8px 12px; align-items:center; justify-content:space-between;
  margin-top:14px; padding-top:14px; border-top:1px solid var(--borde); }
.small-btn { background:#fff; border:1px solid #b9cde1; color:#15477a; border-radius:9px;
  font-weight:650; padding:8px 12px; }
.small-btn:hover { background:#eaf3fe; }
.context { font-size:12px; color:var(--sub); }
.context strong { color:var(--tinta); }
.kpis { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:12px; margin:0 0 20px; }
.kpi { padding:15px 17px; }
.kpi .lab { font-size:11px; text-transform:uppercase; letter-spacing:.05em;
  color:#64758a; font-weight:750; }
.kpi .num { margin:5px 0 0; font-size:clamp(20px,2.2vw,29px); line-height:1.2;
  letter-spacing:-.03em; font-weight:800; font-variant-numeric:tabular-nums; }
.kpi .foot { color:#62778b; font-size:11px; margin-top:5px; }
.section { display:none; }
.section.active { display:block; animation:fadeIn .22s ease; }
@keyframes fadeIn { from {opacity:.65;transform:translateY(4px)} to {opacity:1;transform:none} }
.section-intro { margin:6px 0 18px; }
h2.page-title { margin:0 0 5px; font-size:clamp(21px,2.2vw,28px); letter-spacing:-.025em; }
.section-intro p { margin:0; max-width:920px; color:var(--sub); }
.grid-2 { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }
.panel { padding:18px; min-width:0; margin-bottom:16px; }
.panel h3 { margin:0 0 5px; font-size:16px; letter-spacing:-.01em; }
.panel .desc { margin:0 0 12px; color:var(--sub); font-size:12.7px; }
.chart { height:360px; width:100%; min-width:0; }
.chart.tall { height:435px; }
.chart.wide { height:410px; }
.note { padding:16px 18px; margin-bottom:16px; border-left:4px solid #3277c2; }
.note h3 { margin:0 0 6px; font-size:14px; }
.note p { margin:0; font-size:13.5px; color:#3b5068; }
.note.warning { background:#fffcf8; border-left-color:#d5952a; }
.note.danger { background:var(--alerta-fondo); border-left-color:var(--alerta); }
.note.success { background:#f7fcfb; border-left-color:var(--verde); }
.note + .note { margin-top:0; }
.two-notes { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }
.two-notes .note { margin-bottom:16px; }
.mini-stats { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:9px; }
.mini-stats > div { padding:11px 12px; background:#f4f8fc; border-radius:9px; }
.mini-stats strong { display:block; font-size:18px; }
.mini-stats span { font-size:11px; color:var(--sub); }
.table-wrap { width:100%; max-width:100%; overflow-x:auto; border:1px solid var(--borde);
  border-radius:10px; background:#fff; }
.data-table { border-collapse:collapse; width:100%; font-size:12px; }
.data-table th { background:#eef4fa; color:#24405c; white-space:nowrap; text-align:left;
  position:sticky; top:0; z-index:1; font-weight:750; padding:9px 11px; }
.data-table td { border-top:1px solid #e7edf3; padding:8px 11px; vertical-align:top;
  white-space:nowrap; max-width:460px; }
.data-table tr:nth-child(even) { background:#fbfdff; }
.data-table tr:hover { background:#f0f7ff; }
.data-table .wrap-text { white-space:normal; min-width:250px; }
.status-flag { display:inline-block; border-radius:20px; padding:1px 8px;
  color:#aa2d3b; background:#fff0f2; border:1px solid #ffd7de; font-weight:650; }
.status-ok { color:#277961; }
.muted { color:var(--sub); }
.table-toolbar { display:flex; flex-wrap:wrap; gap:10px; align-items:center; justify-content:space-between; margin:8px 0 12px; }
.table-toolbar input { min-width:220px; max-width:100%; padding:9px 11px;
  border:1px solid #cbd9e6; border-radius:9px; }
.table-toolbar select { max-width:100%; min-width:220px; padding:9px 11px;
  border:1px solid #cbd9e6; border-radius:9px; background:#fff; }
.pager { display:flex; gap:8px; align-items:center; flex-wrap:wrap; justify-content:flex-end;
  margin-top:10px; color:var(--sub); font-size:12px; }
.pager button { border:1px solid var(--borde); background:#fff; border-radius:8px;
  padding:6px 11px; }
.pager button:disabled { opacity:.4; cursor:not-allowed; }
.summary-details { border:1px solid var(--borde); border-radius:10px; background:#fbfdff;
  margin:10px 0 0; }
.summary-details summary { cursor:pointer; padding:13px 16px; font-weight:650; }
.summary-details pre { font:12px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace;
  max-height:360px; overflow:auto; padding:12px 16px; margin:0; border-top:1px solid var(--borde);
  white-space:pre-wrap; overflow-wrap:anywhere; color:#31435c; }
.empty { color:#63748a; background:#f6f9fc; padding:28px; text-align:center;
  border:1px dashed #c4d2e1; border-radius:11px; font-size:13px; }
.footnote { font-size:12px; color:var(--sub); margin:12px 0 0; }
footer { border-top:1px solid var(--borde); margin-top:34px; padding-top:17px;
  font-size:12px; color:var(--sub); }
.offline { display:none; background:#fff5f0; border:1px solid #f1c5b5;
  color:#8a3820; padding:13px 16px; border-radius:10px; margin:0 0 16px; }
.tagline { display:inline-block; padding:3px 9px; border-radius:6px;
  background:#edf4fd; color:#25527f; font-size:11px; font-weight:750; }
@media(max-width:1050px) { .kpis { grid-template-columns:repeat(3,minmax(0,1fr)); }
  .filter-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } }
@media(max-width:720px) { .hero { padding:28px 20px; }
  .shell { margin-top:17px; } .filter-grid { grid-template-columns:1fr; }
  .grid-2,.two-notes { grid-template-columns:1fr; }
  .kpis { grid-template-columns:repeat(2,minmax(0,1fr)); }
  .nav { padding:0 8px; } .tab { padding:14px 13px; font-size:12px; }
  .filters,.panel { padding:15px; } .chart { height:330px; }
  .table-toolbar input,.table-toolbar select { width:100%; }
}
@media(prefers-reduced-motion:reduce) { *,*::before,*::after { animation:none!important; scroll-behavior:auto!important; } }
</style>
</head>
<body>
<a class="skip" href="#contenido">Saltar al contenido</a>
<header class="hero">
  <div class="eyebrow">Ingeniería metalúrgica · Explorador de datos</div>
  <h1>Atlas de aceros SAE</h1>
  <p>Composición química, propiedades mecánicas, tratamientos de procesamiento y confiabilidad de datos. Un análisis visual de aceros al carbono y aleados con criterios metalúrgicos.</p>
  <div class="hero-meta">
    <span id="hero-total">— registros</span>
    <span>UTS · YS · HB · Elongación</span>
    <span id="hero-fecha">Actualizado: —</span>
  </div>
</header>
<nav class="nav-outer" aria-label="Secciones del análisis">
  <div class="nav" role="tablist" aria-label="Análisis metalúrgico">
    <button class="tab" id="tab-carbono" role="tab" aria-selected="true" aria-controls="sec-carbono" tabindex="0" data-tab="carbono">01 · %C vs. Propiedades</button>
    <button class="tab" id="tab-procesos" role="tab" aria-selected="false" aria-controls="sec-procesos" tabindex="-1" data-tab="procesos">02 · Condiciones y Microestructura</button>
    <button class="tab" id="tab-anova" role="tab" aria-selected="false" aria-controls="sec-anova" tabindex="-1" data-tab="anova">03 · Recocido, Normalizado y ANOVA</button>
    <button class="tab" id="tab-aleados" role="tab" aria-selected="false" aria-controls="sec-aleados" tabindex="-1" data-tab="aleados">04 · 10xx vs. Aleados</button>
    <button class="tab" id="tab-auditoria" role="tab" aria-selected="false" aria-controls="sec-auditoria" tabindex="-1" data-tab="auditoria">05 · Auditoría y Datos</button>
  </div>
</nav>
<main id="contenido" class="shell">
  <div id="aviso-cdn" class="offline" role="alert">No se pudo cargar Plotly.js desde la CDN HTTPS. Conéctate a internet para visualizar las gráficas. Los filtros, tablas y textos siguen disponibles.</div>
  <section class="filters" aria-label="Filtros generales compartidos">
    <div class="filter-heading"><h2>Filtros interactivos</h2><span class="helper">Aplicados a las gráficas y métricas dinámicas de las cinco pestañas</span></div>
    <div class="filter-grid">
      <div class="control"><label for="f-condicion">Condición de procesamiento</label><select id="f-condicion"></select></div>
      <div class="control"><label for="f-familia">Familia SAE</label><select id="f-familia"></select></div>
      <div class="control"><label for="f-variable">Propiedad mecánica destacada</label>
        <select id="f-variable"><option value="UTS">Resistencia última (UTS, MPa)</option><option value="YS">Límite elástico (YS, MPa)</option><option value="HB">Dureza Brinell (HB)</option><option value="ELONG">Elongación (%)</option></select></div>
      <div class="control"><label for="f-cmin">Rango de carbono, %C mínimo y máximo</label>
        <div class="range-pair"><input id="f-cmin" type="number" step="0.01" aria-label="Carbono mínimo"><input id="f-cmax" type="number" step="0.01" aria-label="Carbono máximo"></div></div>
      <div class="control"><label for="f-anomalias">Registros con anomalías</label><select id="f-anomalias"><option value="resaltar">Incluir y resaltar</option><option value="excluir">Excluir</option><option value="incluir">Incluir sin resaltar</option></select></div>
      <div class="control"><label for="f-valores">UTS y YS usados en los gráficos</label><select id="f-valores"><option value="corregidos">Conciliados MPa/ksi (script 01)</option><option value="originales">Originales del CSV</option></select></div>
    </div>
    <div class="filter-bottom"><div class="context" id="estado-filtros" aria-live="polite">Preparando datos…</div>
      <button id="limpiar-filtros" class="small-btn" type="button">Restablecer filtros</button></div>
  </section>
  <div class="kpis" aria-label="Indicadores de los registros filtrados">
    <div class="kpi"><div class="lab">Registros visibles</div><div id="kpi-n" class="num">—</div><div class="foot">Filas filtradas</div></div>
    <div class="kpi"><div class="lab">Grados SAE únicos</div><div id="kpi-grados" class="num">—</div><div class="foot">Composición nominal por grado</div></div>
    <div class="kpi"><div class="lab">Carbono medio</div><div id="kpi-carbono" class="num">—</div><div class="foot">% en masa · C_pct</div></div>
    <div class="kpi"><div class="lab" id="kpi-prop-label">UTS media</div><div id="kpi-prop" class="num">—</div><div class="foot" id="kpi-prop-foot">MPa · registros con dato</div></div>
    <div class="kpi"><div class="lab">Registros con anomalía</div><div id="kpi-flags" class="num">—</div><div class="foot">Según la auditoría original</div></div>
  </div>

  <section class="section active" id="sec-carbono" role="tabpanel" aria-labelledby="tab-carbono">
    <div class="section-intro"><h2 class="page-title">Carbono y propiedades mecánicas</h2>
      <p>Relación del contenido representativo de carbono con UTS, YS, dureza Brinell y elongación. Cada punto representa un registro SAE; las rectas y correlaciones se recalculan con los filtros actuales.</p></div>
    <div class="note success"><h3>¿Por qué importa el carbono?</h3><p>En aceros hipoeutectoides enfriados suficientemente despacio suele aumentar la fracción de perlita al aumentar %C: crecen la resistencia y dureza, mientras disminuye la ductilidad. En aceros aleados o templados el resultado depende también de templabilidad, microestructura, tamaño de sección y enfriamiento.</p></div>
    <div class="grid-2">
      <article class="panel"><h3>Resistencia última a tracción (UTS)</h3><p class="desc">MPa · tendencias por familia</p><div id="chart-c-uts" class="chart"></div></article>
      <article class="panel"><h3>Límite elástico (YS)</h3><p class="desc">MPa · tendencias por familia</p><div id="chart-c-ys" class="chart"></div></article>
      <article class="panel"><h3>Dureza Brinell</h3><p class="desc">HB · no se mezclan registros HRB</p><div id="chart-c-hb" class="chart"></div></article>
      <article class="panel"><h3>Elongación</h3><p class="desc">% · ductilidad frente a contenido de carbono</p><div id="chart-c-elong" class="chart"></div></article>
    </div>
    <div class="panel"><h3>Correlaciones calculadas sobre la selección visible</h3>
      <p class="desc">Pearson r, Spearman ρ y pendiente por cada +0.10 %C. No se dibuja recta con n &lt; 5 o un rango de carbono &lt; 0.15 puntos porcentuales; las correlaciones son descriptivas.</p>
      <div id="tabla-correlaciones" class="table-wrap"></div></div>
    <div class="note warning"><h3>Interpretación responsable</h3><p>Una correlación global no demuestra que el carbono sea el único factor. Mezclar laminado, estirado, recocido, normalizado y temple/revenido puede producir tendencias confundidas. El %C del archivo es representativo de límites Min/Max, no una medición real de colada.</p></div>
    <div id="referencia-carbono"></div>
  </section>

  <section class="section" id="sec-procesos" role="tabpanel" aria-labelledby="tab-procesos" hidden>
    <div class="section-intro"><h2 class="page-title">Condiciones y microestructura</h2>
      <p>Distribución de propiedades por ruta de procesamiento y mezcla de familias SAE. El selector de propiedad controla los gráficos de esta pestaña.</p></div>
    <div class="grid-2">
      <article class="panel"><h3 id="titulo-box-cond">Propiedad por condición</h3><p class="desc">Cajas y puntos individuales · mediana y dispersión · condiciones presentes en el filtro</p><div id="chart-procesos-box" class="chart tall"></div></article>
      <article class="panel"><h3>Composición de familias por condición</h3><p class="desc">Número de registros por familia SAE y ruta de procesamiento</p><div id="chart-procesos-familias" class="chart tall"></div></article>
    </div>
    <div class="two-notes">
      <div class="note"><h3>Deformación plástica en frío</h3><p>El estirado en frío aumenta la densidad de dislocaciones y suele elevar YS de manera marcada, con menor elongación. La comparación Hot rolled ↔ Cold drawn es más interpretable cuando se hace sobre el mismo SAE Grade, como en los pares del script 02.</p></div>
      <div class="note"><h3>Procesos térmicos</h3><p>El recocido favorece transformaciones difusionales y microestructuras relativamente blandas; el normalizado, por enfriamiento al aire, puede refinar perlita y grano. El temple seguido de revenido puede producir martensita revenida con precipitación de carburos y una combinación distinta de resistencia y ductilidad.</p></div>
    </div>
    <div class="note warning"><h3>No atribuir las diferencias globales solo al proceso</h3><p>En el conjunto original algunas condiciones térmicas se concentran en aceros aleados y las condiciones de laminado/estirado en familias al carbono. Además, el esferoidizado del conjunto puede incluir estirado posterior, por lo que no representa necesariamente un recocido esferoidizante puro.</p></div>
    <div id="referencia-procesos"></div>
  </section>

  <section class="section" id="sec-anova" role="tabpanel" aria-labelledby="tab-anova" hidden>
    <div class="section-intro"><h2 class="page-title">Recocido, normalizado y ANOVA</h2>
      <p>Exploración de pares del mismo SAE Grade y lectura de los contrastes estadísticos generados por el script 03. Las pruebas precalculadas se identifican por separado de los gráficos recalculados.</p></div>
    <div class="note success"><h3>Comparación emparejada: Annealed vs. Normalized</h3>
      <p>Cuando el mismo grado SAE aparece en ambos tratamientos, comparar las diferencias por grado reduce la confusión por composición nominal. El cambio sigue representando el ciclo térmico completo, incluida la temperatura de tratamiento y la velocidad de enfriamiento.</p></div>
    <div class="grid-2">
      <article class="panel"><h3 id="titulo-pares">Pares del mismo grado SAE</h3><p class="desc">Segmento = mismo grado · recocido y normalizado · selección interactiva</p><div id="chart-pares" class="chart tall"></div></article>
      <article class="panel"><h3>Lectura de pares visibles</h3><p class="desc">Estadísticos descriptivos recalculados para la propiedad seleccionada; sin pruebas p automáticas.</p><div id="pares-metricas" class="mini-stats"></div><p class="footnote">Para los p-valores, supuestos y ajustes Holm debe consultarse el análisis original del script 03, sin filtros adicionales.</p></article>
    </div>
    <div class="panel"><h3>ANOVA del script 03 · resultados originales</h3>
      <p class="desc">La tabla siguiente contiene el análisis precalculado por diseño para la propiedad seleccionada. NO se recalcula al cambiar filtros, modo de anomalías ni columnas corregidas.</p>
      <div id="tabla-anova" class="table-wrap"></div></div>
    <div class="two-notes">
      <div class="note warning"><h3>Supuestos y validez</h3><p>El ANOVA requiere observaciones independientes, normalidad aproximada de residuos y homogeneidad de varianzas; Welch ayuda ante heterocedasticidad. Los valores son nominales de tabla SAE, no réplicas experimentales. Un p pequeño no establece causalidad cuando proceso, carbono y aleación están confundidos.</p></div>
      <div class="note danger"><h3>No todas las comparaciones son legítimas</h3><p>Recocido vs. normalizado comparte grados SAE y requiere un análisis pareado. Incluir esferoidizado con estirado, o condiciones con muestras muy pequeñas, puede volver inválida o no evaluable la atribución de efectos. Consultar el nivel de validez y razones en la tabla original.</p></div>
    </div>
    <div id="referencia-anova"></div>
  </section>

  <section class="section" id="sec-aleados" role="tabpanel" aria-labelledby="tab-aleados" hidden>
    <div class="section-intro"><h2 class="page-title">Aceros 10xx frente a aceros aleados</h2>
      <p>Comparación de aceros al carbono simples (10xx), resulfurados/refosforados (11xx–12xx), alto Mn (13xx–15xx) y aleados 3xxx–9xxx.</p></div>
    <div class="note success"><h3>Fundamento metalúrgico del contraste</h3><p>En 10xx, bajo enfriamientos moderados, predominan ferrita y perlita, con fuerte dependencia de %C. En aceros aleados, Cr, Mn, Ni y Mo pueden retardar transformaciones difusionales y aumentar la templabilidad, permitiendo perlita fina, bainita o martensita según el ciclo. Mn, Ni y Si pueden fortalecer por solución sólida; Cr y Mo pueden intervenir en carburos y precipitación, especialmente tras revenido. La presencia de aleantes por sí sola no garantiza una propiedad superior.</p></div>
    <div class="grid-2">
      <article class="panel"><h3 id="titulo-box-fam">Propiedad por familia</h3><p class="desc">Distribuciones entre las cuatro familias · datos filtrados</p><div id="chart-familias-box" class="chart tall"></div></article>
      <article class="panel"><h3 id="titulo-carbono-fam">Carbono vs. propiedad: 10xx y aleados</h3><p class="desc">Regresiones por familia únicamente con soporte suficiente</p><div id="chart-familias-c" class="chart tall"></div></article>
    </div>
    <div class="panel"><h3>Contraste descriptivo de la selección</h3><p class="desc">Medias de 10xx y 3xxx–9xxx, sin inferir causalidad. Si alguna familia desaparece por los filtros, no existe contraste.</p><div id="contraste-actual"></div></div>
    <div class="note warning"><h3>Condiciones comparables y cobertura química</h3><p>La comparación global mezcla familias con diferente distribución de %C y tratamientos. La ausencia de % de un aleante no equivale a 0 %. Se requieren registros de ambas familias bajo condiciones equivalentes antes de interpretar diferencias como posibles efectos de aleación.</p></div>
    <div id="referencia-aleados"></div>
  </section>

  <section class="section" id="sec-auditoria" role="tabpanel" aria-labelledby="tab-auditoria" hidden>
    <div class="section-intro"><h2 class="page-title">Auditoría, límites y datos explorables</h2>
      <p>Inspección de registros, indicadores de calidad, hallazgos OCR y exportación de la selección. No se modifica el CSV original ni se ocultan anomalías salvo que el usuario lo elija.</p></div>
    <div class="grid-2">
      <article class="panel"><h3>Categorías de hallazgos</h3><p class="desc">Conteo de registros visibles con cada categoría; una fila puede acumular varias.</p><div id="chart-auditoria" class="chart"></div></article>
      <article class="panel"><h3>Notas para interpretar los datos</h3><p class="desc">Cobertura y criterios de auditoría en la muestra seleccionada</p><div id="resumen-auditoria" class="mini-stats"></div></article>
    </div>
    <div class="note danger"><h3>Anomalías que no deben tratarse como mediciones confiables</h3><p>Se auditaron inconsistencias MPa/ksi, casos con YS original superior a UTS original, errores de captura de %C, composición Min &gt; Max y durezas de escala HRB registradas en una columna etiquetada HB. Las correcciones están documentadas y los datos originales se conservan cuando existen columnas separadas.</p></div>
    <div class="panel"><h3>Tabla exploratoria filtrable</h3>
      <p class="desc">Muestra las filas que cumplen los filtros generales. El buscador se aplica dentro de esa selección; la descarga CSV incluye exactamente las filas encontradas.</p>
      <div class="table-toolbar"><input id="f-buscar" type="search" placeholder="Buscar SAE, proceso, anomalía…" aria-label="Buscar registros"><button class="small-btn" id="exportar-csv" type="button">Descargar selección CSV</button></div>
      <div id="tabla-explorador" class="table-wrap"></div><div class="pager"><span id="paginas-texto"></span><button id="pag-prev" type="button">Anterior</button><button id="pag-next" type="button">Siguiente</button></div></div>
    <div class="note warning"><h3>Limitaciones metodológicas</h3><p>Los datos son nominales de tablas de propiedades por grado y condición, no ensayos repetidos. El tamaño de muestra entre familias es desigual y varios procesos no se solapan. Los valores de %C provienen de rangos especificados; las regresiones son descriptivas. No se conocen tamaño de barra, velocidades reales de enfriamiento ni porcentaje de deformación en frío.</p></div>
    <div id="referencia-auditoria"></div>
  </section>
  <footer>
    <strong>Atlas de aceros SAE</strong> · Datos integrados al generar este HTML; no se envían a servidores.
    Gráficos mediante Plotly.js (CDN HTTPS). Para publicar, sube el HTML a GitHub Pages (raíz o /docs).
    Los filtros recalculan estadísticas descriptivas, no las pruebas ANOVA originales.
  </footer>
</main>
<script id="datos-integrados" type="application/json">__DATOS_JSON__</script>
<script>
(function () {
  "use strict";
  const paquete = JSON.parse(document.getElementById("datos-integrados").textContent);
  const registros = paquete.registros || [];
  const tablas = paquete.tablas || {};
  const summaries = paquete.resumenes || {};
  const meta = paquete.meta || {};
  const FAMILIAS = ["Carbono simple 10xx", "Resulfurados/Refosforados 11xx-12xx", "Alto Mn 13xx-15xx", "Aleados 3xxx-9xxx"];
  const ABREV = {"Carbono simple 10xx":"10xx", "Resulfurados/Refosforados 11xx-12xx":"11xx–12xx", "Alto Mn 13xx-15xx":"13xx–15xx", "Aleados 3xxx-9xxx":"3xxx–9xxx"};
  const COLORES = {"Carbono simple 10xx":"#1872be", "Resulfurados/Refosforados 11xx-12xx":"#e49c22", "Alto Mn 13xx-15xx":"#128977", "Aleados 3xxx-9xxx":"#d65a45"};
  const OTRO = "#79879b", ALERTA = "#c72e47";
  const CONDICIONES = ["Hot rolled", "Cold drawn", "Annealed", "Annealed + Cold drawn", "Spheroidized annealed", "Normalized", "Normalized + Cold drawn", "Quenched & tempered", "Other"];
  const METRICAS = {
    UTS: {nombre:"UTS", unidad:"MPa", etiqueta:"Resistencia última a la tracción"},
    YS: {nombre:"YS", unidad:"MPa", etiqueta:"Límite elástico"},
    HB: {nombre:"Dureza", unidad:"HB", etiqueta:"Dureza Brinell"},
    ELONG: {nombre:"Elongación", unidad:"%", etiqueta:"Elongación"}
  };
  const $ = id => document.getElementById(id);
  const N = v => v !== null && v !== undefined && v !== "" && Number.isFinite(Number(v)) ? Number(v) : null;
  const fmt = (v, d=1) => N(v) === null ? "—" : Number(v).toLocaleString("es-MX", {maximumFractionDigits:d, minimumFractionDigits:d});
  const compact = (v, d=2) => N(v) === null ? "—" : Number(v).toLocaleString("es-MX", {maximumFractionDigits:d});
  const signed = (v, d=1) => N(v) === null ? "—" : `${v>=0?"+":""}${fmt(v,d)}`;
  const esc = v => String(v ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"})[c]);
  const rowNum = (r, col) => N(r[col]);
  const varCol = (clave, modo) => clave === "UTS" ? (modo === "corregidos" ? "UTS_MPa_corr":"UTS (MPa)")
    : clave === "YS" ? (modo === "corregidos" ? "YS_MPa_corr":"YS (MPa)")
    : clave === "HB" ? "Hardness_HB_num" : "Elongation (%)";
  const mean = a => a.length ? a.reduce((x,y)=>x+y,0)/a.length : null;
  const median = a => { if(!a.length) return null; const b=[...a].sort((x,y)=>x-y); const h=Math.floor(b.length/2); return b.length%2?b[h]:(b[h-1]+b[h])/2; };
  const stdev = a => a.length>1 ? Math.sqrt(a.reduce((s,x)=>s+(x-mean(a))**2,0)/(a.length-1)):null;
  const uniq = a => [...new Set(a)];
  let active="carbono", page=0;
  const PAGE=16;
  const initial={condicion:"Todas",familia:"Todas",variable:"UTS",cmin:meta.carbono_min,cmax:meta.carbono_max,
    anomalias:meta.modo_anomalias||"resaltar",valores:meta.modo_valores||"corregidos"};
  let filtros={...initial};

  function regression(rows, clave) {
    const pares=rows.map(r=>[rowNum(r,"C_pct"),rowNum(r,varCol(clave,filtros.valores))]).filter(p=>p[0]!==null&&p[1]!==null);
    const n=pares.length, xs=pares.map(p=>p[0]), ys=pares.map(p=>p[1]);
    const rango=n ? Math.max(...xs)-Math.min(...xs) : 0;
    if(n<3) return {n,rango,r:null,rho:null,m:null,b:null,r2:null,valido:false};
    const mx=mean(xs), my=mean(ys);
    let sxx=0, syy=0, sxy=0;
    for(let i=0;i<n;i++) { const dx=xs[i]-mx,dy=ys[i]-my; sxx+=dx*dx;syy+=dy*dy;sxy+=dx*dy; }
    const m=sxx>0 ? sxy/sxx:null, b=m!==null ? my-m*mx:null;
    const r=sxx>0&&syy>0 ? sxy/Math.sqrt(sxx*syy):null;
    function ranks(a) {
      const ord=a.map((v,i)=>({v,i})).sort((a,b)=>a.v-b.v), result=new Array(a.length);
      for(let i=0;i<ord.length;) {let j=i+1;while(j<ord.length&&ord[j].v===ord[i].v)j++;const rango=(i+j+1)/2;for(let k=i;k<j;k++)result[ord[k].i]=rango;i=j;}
      return result;
    }
    const rx=ranks(xs), ry=ranks(ys), av=mean(rx), bv=mean(ry);
    let xx=0,yy=0,xy=0;
    for(let i=0;i<n;i++){xx+=(rx[i]-av)**2;yy+=(ry[i]-bv)**2;xy+=(rx[i]-av)*(ry[i]-bv);}
    const rho=xx>0&&yy>0 ? xy/Math.sqrt(xx*yy):null;
    return {n,rango,r,rho,m,b,r2:r===null?null:r*r,valido:n>=5&&rango>=.15&&m!==null};
  }

  function optsFor(id, values) {
    const sel=$(id); sel.innerHTML="";
    for(const [value,label] of values) {const op=document.createElement("option");op.value=value;op.textContent=label;sel.add(op);}
  }
  function setup() {
    optsFor("f-condicion",[["Todas","Todas las condiciones"],...uniq(registros.map(r=>r.Condicion_Grupo)).sort((a,b)=>CONDICIONES.indexOf(a)-CONDICIONES.indexOf(b)).map(g=>[g,g])]);
    optsFor("f-familia",[["Todas","Todas las familias"],...uniq(registros.map(r=>r.Familia_SAE)).sort((a,b)=>FAMILIAS.indexOf(a)-FAMILIAS.indexOf(b)).map(f=>[f,ABREV[f]?`${ABREV[f]} · ${f}`:f])]);
    $("f-cmin").value=Number(meta.carbono_min).toFixed(2);
    $("f-cmax").value=Number(meta.carbono_max).toFixed(2);
    $("f-cmin").min=(Number(meta.carbono_min)-.01).toFixed(2);
    $("f-cmax").max=(Number(meta.carbono_max)+.01).toFixed(2);
    $("f-anomalias").value=filtros.anomalias;
    $("f-valores").value=filtros.valores;
    $("hero-total").textContent=`${registros.length} registros`;
    $("hero-fecha").textContent=`Generado: ${(meta.fecha_generacion||"").slice(0,10)}`;
    document.querySelectorAll(".tab").forEach(btn=>{
      btn.addEventListener("click",()=>changeTab(btn.dataset.tab));
      btn.addEventListener("keydown",e=>{
        if(e.key!=="ArrowLeft"&&e.key!=="ArrowRight")return;
        e.preventDefault();const tabs=[...document.querySelectorAll(".tab")],idx=tabs.indexOf(btn);
        const next=tabs[(idx+(e.key==="ArrowRight"?1:-1)+tabs.length)%tabs.length];next.focus();changeTab(next.dataset.tab);
      });
    });
    for(const id of ["f-condicion","f-familia","f-variable","f-anomalias","f-valores","f-cmin","f-cmax"]) {
      $(id).addEventListener("change",updateControls);
    }
    $("limpiar-filtros").addEventListener("click",()=>{filtros={...initial};applyControls();page=0;refresh();});
    $("f-buscar").addEventListener("input",()=>{page=0;if(active==="auditoria")renderAuditoria();});
    $("pag-prev").addEventListener("click",()=>{page=Math.max(0,page-1);renderExplorer();});
    $("pag-next").addEventListener("click",()=>{page++;renderExplorer();});
    $("exportar-csv").addEventListener("click",exportCSV);
    for(const section of ["carbono","procesos","anova","aleados","auditoria"]) setupReferences(section);
    setupSummaries();
    refresh();
    if (!window.Plotly) $("aviso-cdn").style.display="block";
  }
  function applyControls() {
    $("f-condicion").value=filtros.condicion;$("f-familia").value=filtros.familia;
    $("f-variable").value=filtros.variable;$("f-anomalias").value=filtros.anomalias;
    $("f-valores").value=filtros.valores;$("f-cmin").value=Number(filtros.cmin).toFixed(2);
    $("f-cmax").value=Number(filtros.cmax).toFixed(2);
  }
  function updateControls() {
    filtros.condicion=$("f-condicion").value;filtros.familia=$("f-familia").value;
    filtros.variable=$("f-variable").value;filtros.anomalias=$("f-anomalias").value;
    filtros.valores=$("f-valores").value;
    const lo=N($("f-cmin").value),hi=N($("f-cmax").value);
    filtros.cmin=lo===null?meta.carbono_min:lo;
    filtros.cmax=hi===null?meta.carbono_max:hi;
    if(filtros.cmin>filtros.cmax) {
      if(document.activeElement===$("f-cmin"))filtros.cmax=filtros.cmin;
      else filtros.cmin=filtros.cmax;
      applyControls();
    }
    page=0;refresh();
  }
  function filtered() {
    return registros.filter(r=>{
      const c=rowNum(r,"C_pct");
      return (filtros.condicion==="Todas"||r.Condicion_Grupo===filtros.condicion)
        && (filtros.familia==="Todas"||r.Familia_SAE===filtros.familia)
        && (c!==null&&c>=filtros.cmin-1e-9&&c<=filtros.cmax+1e-9)
        && (filtros.anomalias!=="excluir"||!r.flag_anomalia);
    });
  }
  function selectedValues(rows,clave) {
    return rows.map(r=>rowNum(r,varCol(clave,filtros.valores))).filter(v=>v!==null);
  }
  function metrics(rows) {
    const vs=selectedValues(rows,filtros.variable),mc=rows.map(r=>rowNum(r,"C_pct")).filter(v=>v!==null);
    $("kpi-n").textContent=rows.length.toLocaleString("es-MX");
    $("kpi-grados").textContent=uniq(rows.map(r=>r["SAE Grade"])).length.toLocaleString("es-MX");
    $("kpi-carbono").textContent=mean(mc)===null?"—":`${fmt(mean(mc),3)} %`;
    $("kpi-prop-label").textContent=`${METRICAS[filtros.variable].nombre} media`;
    $("kpi-prop").textContent=mean(vs)===null?"—":`${fmt(mean(vs),1)} ${METRICAS[filtros.variable].unidad}`;
    $("kpi-prop-foot").textContent=`${vs.length} registros con dato válido`;
    $("kpi-flags").textContent=rows.filter(r=>r.flag_anomalia).length;
    $("estado-filtros").innerHTML=`<strong>${rows.length}</strong> de ${registros.length} registros · ${esc(filtros.condicion)} · ${esc(filtros.familia)} · ${fmt(filtros.cmin,2)}–${fmt(filtros.cmax,2)} %C`;
  }
  function changeTab(name) {
    active=name;
    for(const btn of document.querySelectorAll(".tab")) {
      const yes=btn.dataset.tab===name;btn.setAttribute("aria-selected",String(yes));btn.tabIndex=yes?0:-1;
    }
    for(const panel of document.querySelectorAll(".section")) {
      const yes=panel.id===`sec-${name}`;panel.hidden=!yes;panel.classList.toggle("active",yes);
    }
    requestAnimationFrame(refresh);
  }
  function refresh() {
    const rows=filtered();metrics(rows);
    if(active==="carbono")renderCarbono(rows);
    else if(active==="procesos")renderProcesos(rows);
    else if(active==="anova")renderAnova(rows);
    else if(active==="aleados")renderAleados(rows);
    else renderAuditoria(rows);
  }
  const baseLayout=(titleX,titleY,more={})=>({
    paper_bgcolor:"#fff",plot_bgcolor:"#fff",font:{family:"system-ui, Segoe UI, sans-serif",color:"#31445d",size:11},
    margin:{l:64,r:22,t:22,b:60},hovermode:"closest",showlegend:true,
    legend:{orientation:"h",x:0,y:-.24,font:{size:10}},
    xaxis:{title:titleX,gridcolor:"#eaf0f6",zeroline:false,automargin:true},
    yaxis:{title:titleY,gridcolor:"#eaf0f6",zeroline:false,automargin:true},...more
  });
  function plot(id,traces,layout) {
    const el=$(id);
    if(!traces.length || traces.every(t=>!t.x?.length && !t.y?.length)) {
      if(window.Plotly && el.classList.contains("js-plotly-plot"))Plotly.purge(el);
      el.innerHTML='<div class="empty">No hay suficientes datos para esta visualización con los filtros actuales.</div>';
      return;
    }
    if(!window.Plotly) {el.innerHTML='<div class="empty">Gráfico no disponible: Plotly.js requiere conexión a internet.</div>';return;}
    Plotly.react(el,traces,layout,{responsive:true,displaylogo:false,scrollZoom:false,
      modeBarButtonsToRemove:["lasso2d","select2d","autoScale2d","toggleSpikelines"]});
  }
  function hoverRows(rows,clave) {
    return rows.map(r=>`${esc(r["SAE Grade"])} · ${esc(r.Condicion_Grupo)}<br>%C: ${compact(rowNum(r,"C_pct"),3)}<br>${METRICAS[clave].nombre}: ${compact(rowNum(r,varCol(clave,filtros.valores)),2)} ${METRICAS[clave].unidad}${r.flag_anomalia?"<br>⚠ Registro marcado":""}`);
  }
  function scatterC(rows,clave,id,principales=false) {
    const use=principales ? FAMILIAS.filter(f=>f===FAMILIAS[0]||f===FAMILIAS[3]):FAMILIAS;
    const trazas=[];
    for(const fam of use) {
      const sub=rows.filter(r=>r.Familia_SAE===fam&&rowNum(r,"C_pct")!==null&&rowNum(r,varCol(clave,filtros.valores))!==null);
      if(!sub.length)continue;
      const e=regression(sub,clave);
      trazas.push({type:"scatter",mode:"markers",name:ABREV[fam],x:sub.map(r=>rowNum(r,"C_pct")),
        y:sub.map(r=>rowNum(r,varCol(clave,filtros.valores))),text:hoverRows(sub,clave),hovertemplate:"%{text}<extra></extra>",
        marker:{size:9,color:COLORES[fam],opacity:.8,line:{color:"#fff",width:.7}}});
      if(e.valido) {
        const xs=[Math.min(...sub.map(r=>rowNum(r,"C_pct"))),Math.max(...sub.map(r=>rowNum(r,"C_pct")))];
        trazas.push({type:"scatter",mode:"lines",name:`${ABREV[fam]} · r=${e.r.toFixed(2)}`,x:xs,
          y:xs.map(x=>e.m*x+e.b),hoverinfo:"skip",line:{color:COLORES[fam],width:2,dash:"dash"},showlegend:true});
      }
    }
    const anomalos=rows.filter(r=>r.flag_anomalia&&rowNum(r,"C_pct")!==null&&rowNum(r,varCol(clave,filtros.valores))!==null
      && (!principales||use.includes(r.Familia_SAE)));
    if(filtros.anomalias==="resaltar"&&anomalos.length) trazas.push({type:"scatter",mode:"markers",name:"Anomalía",
      x:anomalos.map(r=>rowNum(r,"C_pct")),y:anomalos.map(r=>rowNum(r,varCol(clave,filtros.valores))),
      text:hoverRows(anomalos,clave),hovertemplate:"%{text}<extra></extra>",
      marker:{size:17,color:"rgba(0,0,0,0)",line:{color:ALERTA,width:2.2},symbol:"circle-open"}});
    plot(id,trazas,baseLayout("Carbono representativo (% masa)",`${METRICAS[clave].nombre} (${METRICAS[clave].unidad})`));
  }
  function makeTable(host,columns,rows,maxRows) {
    if(!rows||!rows.length) {$(host).innerHTML='<div class="empty">Sin registros para mostrar.</div>';return;}
    const cols=columns&&columns.length?columns:Object.keys(rows[0]);
    const visible=maxRows?rows.slice(0,maxRows):rows;
    $(host).innerHTML=`<table class="data-table"><thead><tr>${cols.map(c=>`<th scope="col">${esc(c)}</th>`).join("")}</tr></thead><tbody>${visible.map(r=>`<tr>${cols.map(c=>{
      const v=r[c];if(c==="flag_anomalia")return `<td>${v?'<span class="status-flag">Sí</span>':'<span class="status-ok">No</span>'}</td>`;
      return `<td class="${String(c).toLowerCase().includes("motivo")||String(c).toLowerCase().includes("nota")?"wrap-text":""}">${esc(typeof v==="number"?compact(v,4):v??"—")}</td>`;
    }).join("")}</tr>`).join("")}</tbody></table>`;
  }
  function renderCarbono(rows) {
    for(const clave of ["UTS","YS","HB","ELONG"])scatterC(rows,clave,`chart-c-${clave.toLowerCase()}`);
    const cells=[];
    for(const fam of FAMILIAS) {
      const s=rows.filter(r=>r.Familia_SAE===fam);
      for(const clave of ["UTS","YS","HB","ELONG"]) {
        if(!s.length)continue;
        const e=regression(s,clave);
        cells.push({"Familia":ABREV[fam],"Propiedad":`${METRICAS[clave].nombre} (${METRICAS[clave].unidad})`,
          "n":e.n,"r Pearson":e.r,"ρ Spearman":e.rho,
          "Pendiente / +0.1 %C":e.valido?e.m*.1:null,"R²":e.r2,
          "Interpretación":e.valido?"Descriptiva":"n o rango %C insuficiente"});
      }
    }
    makeTable("tabla-correlaciones",["Familia","Propiedad","n","r Pearson","ρ Spearman","Pendiente / +0.1 %C","R²","Interpretación"],cells);
  }
  function conditionNames(rows) {
    const pres=new Set(rows.map(r=>r.Condicion_Grupo));
    return [...CONDICIONES.filter(c=>pres.has(c)),...Array.from(pres).filter(c=>!CONDICIONES.includes(c)).sort()];
  }
  function boxBy(rows,clave,id,tipo) {
    const grupos=tipo==="familia"?FAMILIAS.filter(f=>rows.some(r=>r.Familia_SAE===f)):conditionNames(rows);
    const campo=tipo==="familia"?"Familia_SAE":"Condicion_Grupo";
    const trazas=[];
    for(const g of grupos) {
      const sub=rows.filter(r=>r[campo]===g&&rowNum(r,varCol(clave,filtros.valores))!==null);
      if(!sub.length)continue;
      const name=tipo==="familia"?ABREV[g]:g;
      const color=tipo==="familia"?COLORES[g]:COLORES[sub[0].Familia_SAE]||"#5883b6";
      trazas.push({type:"box",name,legendgroup:name,showlegend:false,boxpoints:"all",jitter:.24,pointpos:0,
        x:sub.map(()=>name),y:sub.map(r=>rowNum(r,varCol(clave,filtros.valores))),text:hoverRows(sub,clave),
        hovertemplate:"%{text}<extra></extra>",fillcolor:`rgba(${parseInt(color.slice(1,3),16)},${parseInt(color.slice(3,5),16)},${parseInt(color.slice(5,7),16)},0.18)`,line:{color,width:1.6},
        marker:{size:7,color,opacity:.75},boxmean:true});
      if(filtros.anomalias==="resaltar") {
        const bad=sub.filter(r=>r.flag_anomalia);
        if(bad.length)trazas.push({type:"scatter",mode:"markers",name:"Anomalía",showlegend:false,
          x:bad.map(()=>name),y:bad.map(r=>rowNum(r,varCol(clave,filtros.valores))),text:hoverRows(bad,clave),
          hovertemplate:"%{text}<extra></extra>",marker:{symbol:"circle-open",size:15,color:ALERTA,line:{width:2}}});
      }
    }
    const lay=baseLayout("",`${METRICAS[clave].nombre} (${METRICAS[clave].unidad})`,{
      showlegend:false,margin:{l:62,r:15,t:16,b:145},
      xaxis:{type:"category",categoryorder:"array",categoryarray:grupos.map(g=>tipo==="familia"?ABREV[g]:g),tickangle:-35,automargin:true}
    });
    plot(id,trazas,lay);
  }
  function renderProcesos(rows) {
    const k=filtros.variable;
    $("titulo-box-cond").textContent=`${METRICAS[k].etiqueta} por condición (${METRICAS[k].unidad})`;
    boxBy(rows,k,"chart-procesos-box","condicion");
    const names=conditionNames(rows);
    const trazas=FAMILIAS.map(f=>({type:"bar",orientation:"h",name:ABREV[f],marker:{color:COLORES[f]},
      x:names.map(c=>rows.filter(r=>r.Condicion_Grupo===c&&r.Familia_SAE===f).length),y:names,hovertemplate:"%{y} · %{x} registros<extra>%{fullData.name}</extra>"}));
    plot("chart-procesos-familias",trazas,baseLayout("Número de registros","",{
      barmode:"stack",margin:{l:155,r:18,t:15,b:65},yaxis:{autorange:"reversed",automargin:true},
      legend:{orientation:"h",x:0,y:-.25,font:{size:10}}}));
  }
  function pairedData(rows,clave) {
    const a=new Map(),b=new Map();
    for(const r of rows) {
      const grado=String(r["SAE Grade"]);
      if(r.Condicion_Grupo==="Annealed" && !a.has(grado))a.set(grado,r);
      if(r.Condicion_Grupo==="Normalized" && !b.has(grado))b.set(grado,r);
    }
    return [...a].filter(([g,r])=>b.has(g)&&rowNum(r,varCol(clave,filtros.valores))!==null&&rowNum(b.get(g),varCol(clave,filtros.valores))!==null)
      .map(([g,r])=>({grado:g,a:rowNum(r,varCol(clave,filtros.valores)),b:rowNum(b.get(g),varCol(clave,filtros.valores)),
        c:rowNum(r,"C_pct"),bad:r.flag_anomalia||b.get(g).flag_anomalia})).sort((x,y)=>(x.c??0)-(y.c??0));
  }
  function renderAnova(rows) {
    const k=filtros.variable,pares=pairedData(rows,k);
    $("titulo-pares").textContent=`Annealed → Normalized · ${METRICAS[k].nombre} (${METRICAS[k].unidad})`;
    const cat=pares.map(p=>p.grado),ys=pares.map(p=>p.b-p.a);
    const lines=[],pointsA=[],pointsB=[];
    for(const p of pares) {
      lines.push({type:"scatter",mode:"lines",x:[p.a,p.b],y:[p.grado,p.grado],showlegend:false,hoverinfo:"skip",line:{color:"#a2afbd",width:2}});
    }
    pointsA.push({type:"scatter",mode:"markers",name:"Recocido",x:pares.map(p=>p.a),y:cat,
      text:pares.map(p=>`${p.grado} · Recocido: ${fmt(p.a)} ${METRICAS[k].unidad}`),hovertemplate:"%{text}<extra></extra>",
      marker:{color:"#17866f",size:9}});
    pointsB.push({type:"scatter",mode:"markers",name:"Normalizado",x:pares.map(p=>p.b),y:cat,
      text:pares.map(p=>`${p.grado} · Normalizado: ${fmt(p.b)} ${METRICAS[k].unidad}<br>Δ: ${signed(p.b-p.a)}`),
      hovertemplate:"%{text}<extra></extra>",marker:{color:"#d6643d",size:9}});
    if(filtros.anomalias==="resaltar" && pares.some(p=>p.bad))pointsB.push({type:"scatter",mode:"markers",
      name:"Par con anomalía",x:pares.filter(p=>p.bad).map(p=>p.b),y:pares.filter(p=>p.bad).map(p=>p.grado),
      marker:{symbol:"circle-open",size:16,color:ALERTA,line:{width:2}}});
    const chart=$("chart-pares");chart.style.height=`${Math.max(360,Math.min(650,150+pares.length*22))}px`;
    plot("chart-pares",pares.length?[...lines,...pointsA,...pointsB]:[],baseLayout(`${METRICAS[k].nombre} (${METRICAS[k].unidad})`,"Grado SAE",{
      margin:{l:90,r:15,t:12,b:68},yaxis:{type:"category",categoryorder:"array",categoryarray:[...cat].reverse(),automargin:true},
      legend:{orientation:"h",y:-.2}}));
    const deltas=ys;
    $("pares-metricas").innerHTML=[
      ["Pares con dato",pares.length],
      ["Δ medio (normalizado − recocido)",deltas.length?`${signed(mean(deltas))} ${METRICAS[k].unidad}`:"—"],
      ["Δ mediana",deltas.length?`${signed(median(deltas))} ${METRICAS[k].unidad}`:"—"],
      ["Pares con aumento",deltas.filter(x=>x>0).length],
      ["Pares con disminución",deltas.filter(x=>x<0).length],
      ["Desviación est. de Δ",deltas.length>1?`${fmt(stdev(deltas))} ${METRICAS[k].unidad}`:"—"]
    ].map(([l,v])=>`<div><strong>${esc(v)}</strong><span>${esc(l)}</span></div>`).join("");
    const origen=tablas["03_anova"];
    if(origen) {
      const filas=(origen.filas||[]).filter(r=>r.clave===k).map(r=>({
        "Diseño":r.diseno_id||"—","Comparación":r.diseno||"—","N":r.N,"Prueba":r.prueba_recomendada||"—",
        "p Holm":r.p_holm,"η²":r.eta2,"Normalidad":r.normalidad_ok,"Levene p":r.levene_p,
        "Validez":r.nivel_validez||"—","Razones":r.razones_validez||"—"
      }));
      makeTable("tabla-anova",["Diseño","Comparación","N","Prueba","p Holm","η²","Normalidad","Levene p","Validez","Razones"],filas);
    } else $("tabla-anova").innerHTML='<div class="empty">No se encontró tabla_03_anova_supuestos.csv. Ejecuta el script 03 para incorporar los resultados estadísticos.</div>';
  }
  function renderAleados(rows) {
    const k=filtros.variable;
    $("titulo-box-fam").textContent=`${METRICAS[k].nombre} por familia (${METRICAS[k].unidad})`;
    $("titulo-carbono-fam").textContent=`%C vs. ${METRICAS[k].nombre}: 10xx / aleados`;
    boxBy(rows,k,"chart-familias-box","familia");
    scatterC(rows,k,"chart-familias-c",true);
    const a=selectedValues(rows.filter(r=>r.Familia_SAE===FAMILIAS[0]),k),b=selectedValues(rows.filter(r=>r.Familia_SAE===FAMILIAS[3]),k);
    const host=$("contraste-actual");
    if(!a.length||!b.length) {host.innerHTML='<div class="empty">No hay observaciones válidas de ambas familias bajo estos filtros. No es posible comparar sus medias.</div>';return;}
    const ma=mean(a),mb=mean(b),delta=mb-ma;
    host.innerHTML=`<div class="mini-stats"><div><strong>${fmt(ma)} ${esc(METRICAS[k].unidad)}</strong><span>10xx · n=${a.length}</span></div>
      <div><strong>${fmt(mb)} ${esc(METRICAS[k].unidad)}</strong><span>Aleados 3xxx–9xxx · n=${b.length}</span></div>
      <div><strong>${signed(delta)} ${esc(METRICAS[k].unidad)}</strong><span>Diferencia de medias: aleados − 10xx</span></div>
      <div><strong>${ma?`${signed(delta/ma*100)} %`:"—"}</strong><span>Diferencia relativa respecto a 10xx (descriptiva)</span></div></div>`;
  }
  function hallazgos(rows) {
    const mapa=new Map();const re=/\[(ANOMALIA|CORRECCION|AVISO):([A-Z0-9_]+)\]/g;
    for(const r of rows) {
      const vistos=new Set();let m;
      while((m=re.exec(String(r.motivo_auditoria||"")))!==null)vistos.add(`${m[1]}:${m[2]}`);
      for(const cat of vistos)mapa.set(cat,(mapa.get(cat)||0)+1);
      re.lastIndex=0;
    }
    return [...mapa].map(([cat,n])=>({cat,n})).sort((a,b)=>b.n-a.n);
  }
  function visibleExplorerRows() {
    const q=($("f-buscar").value||"").trim().toLocaleLowerCase("es");
    const rows=filtered();
    if(!q)return rows;
    return rows.filter(r=>[r["SAE Grade"],r.Familia_SAE,r.Condicion_Grupo,r.Conditions_limpia,r.motivo_auditoria,r.registro]
      .some(s=>String(s??"").toLocaleLowerCase("es").includes(q)));
  }
  function renderExplorer() {
    const rows=visibleExplorerRows(),totalPages=Math.max(1,Math.ceil(rows.length/PAGE));
    page=Math.min(page,totalPages-1);const slice=rows.slice(page*PAGE,(page+1)*PAGE);
    const c=varCol(filtros.variable,filtros.valores);
    const columnas=["registro","SAE Grade","Familia_SAE","Condicion_Grupo","C_pct",c,"Hardness_HB_num","Elongation (%)","flag_anomalia","motivo_auditoria"];
    makeTable("tabla-explorador",columnas,slice);
    $("paginas-texto").textContent=`${rows.length} filas · página ${page+1} de ${totalPages}`;
    $("pag-prev").disabled=page<=0;$("pag-next").disabled=page>=totalPages-1;
  }
  function renderAuditoria(inputRows) {
    const rows=inputRows||filtered(),res=hallazgos(rows);
    const top=res.slice(0,13).reverse();
    const trazas=top.length?[{type:"bar",orientation:"h",y:top.map(x=>x.cat),x:top.map(x=>x.n),marker:{color:top.map(x=>x.cat.startsWith("ANOMALIA")?ALERTA:x.cat.startsWith("CORRECCION")?"#2f76c2":"#a3b0c1")},hovertemplate:"%{y}: %{x} registros<extra></extra>"}]:[];
    plot("chart-auditoria",trazas,baseLayout("Número de registros","",{
      margin:{l:205,r:18,t:14,b:52},showlegend:false,yaxis:{automargin:true}}));
    const flagged=rows.filter(r=>r.flag_anomalia).length,inc=rows.filter(r=>String(r.motivo_auditoria||"").trim()!=="").length;
    const hbc=rows.filter(r=>r.Hardness_escala&&r.Hardness_escala!=="HB").length;
    const validos=rows.filter(r=>rowNum(r,"Hardness_HB_num")!==null).length;
    $("resumen-auditoria").innerHTML=[
      ["Registros seleccionados",rows.length],
      ["Con al menos un hallazgo",inc],
      ["Con flag_anomalia",flagged],
      ["Dureza HB disponible",validos],
      ["Escala de dureza distinta a HB",hbc],
      ["Tipos/categorías detectadas",res.length]
    ].map(([l,v])=>`<div><strong>${esc(v)}</strong><span>${esc(l)}</span></div>`).join("");
    renderExplorer();
  }
  function exportCSV() {
    const rows=visibleExplorerRows();
    if(!rows.length) {alert("No hay registros que exportar con los filtros actuales.");return;}
    const cols=["registro","SAE Grade","Familia_SAE","Condicion_Grupo","C_pct","UTS (MPa)","YS (MPa)",
      "UTS_MPa_corr","YS_MPa_corr","Hardness_HB_num","Elongation (%)","flag_anomalia","motivo_auditoria"];
    const csv=v=>`"${String(v??"").replace(/"/g,'""')}"`;
    const texto=[cols.map(csv).join(","),...rows.map(r=>cols.map(k=>csv(r[k])).join(","))].join("\r\n");
    const blob=new Blob(["\ufeff",texto],{type:"text/csv;charset=utf-8;"});
    const url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;a.download="aceros_SAE_filtrados.csv";
    document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  function setupReferences(section) {
    const host=$("referencia-"+section);
    const items=Object.entries(tablas).filter(([,v])=>v.seccion===section);
    if(!items.length) {
      host.innerHTML='<div class="note"><h3>Tablas complementarias opcionales</h3><p>Esta sección funciona con el CSV limpio. Ejecuta los scripts anteriores para integrar aquí sus tablas de referencia.</p></div>';
      return;
    }
    const title=section==="auditoria"?"Tablas originales de auditoría y sensibilidad":"Tablas de referencia de los scripts anteriores";
    host.innerHTML=`<div class="panel"><h3>${title}</h3><p class="desc"><strong>Resultados precalculados.</strong> No cambian con los filtros actuales. Se muestran para trazabilidad; consulta las limitaciones de los scripts de origen.</p>
      <div class="table-toolbar"><select id="ref-select-${section}" aria-label="Elegir tabla de referencia"></select>
      <span class="context" id="ref-caption-${section}"></span></div><div id="ref-table-${section}" class="table-wrap"></div>
      <div class="pager"><span id="ref-page-text-${section}"></span><button id="ref-prev-${section}" type="button">Anterior</button><button id="ref-next-${section}" type="button">Siguiente</button></div></div>`;
    const select=$("ref-select-"+section);
    for(const [clave,t] of items){const opt=document.createElement("option");opt.value=clave;opt.textContent=t.nombre;select.add(opt);}
    let refpage=0;
    function draw(){
      const tabla=tablas[select.value],total=tabla.filas.length,max=Math.max(1,Math.ceil(total/PAGE));
      refpage=Math.min(refpage,max-1);
      makeTable("ref-table-"+section,tabla.columnas,tabla.filas.slice(refpage*PAGE,(refpage+1)*PAGE));
      $("ref-caption-"+section).textContent=tabla.ruta;
      $("ref-page-text-"+section).textContent=`${total} filas · página ${refpage+1} de ${max}`;
      $("ref-prev-"+section).disabled=refpage<=0;$("ref-next-"+section).disabled=refpage>=max-1;
    }
    select.addEventListener("change",()=>{refpage=0;draw();});
    $("ref-prev-"+section).addEventListener("click",()=>{refpage--;draw();});
    $("ref-next-"+section).addEventListener("click",()=>{refpage++;draw();});
    draw();
  }
  function setupSummaries() {
    for(const section of ["carbono","anova","aleados"]) {
      if(!summaries[section])continue;
      const host=$("referencia-"+section),details=document.createElement("details");
      details.className="summary-details";
      const label=document.createElement("summary");label.textContent="Ver el resumen técnico íntegro del script "+({carbono:"02",anova:"03",aleados:"04"}[section]);
      const pre=document.createElement("pre");pre.textContent=summaries[section];
      details.append(label,pre);host.append(details);
    }
  }
  // El archivo funciona desde file://: los datos NO se descargan con fetch().
  // La CDN es diferida; setup se ejecuta cuando ya se ha procesado todo el HTML.
  if(document.readyState==="loading")document.addEventListener("DOMContentLoaded",setup);
  else setup();
  // La CDN Plotly.js llega con defer. Si aún no llegó al montar el dashboard,
  // el evento load repinta la pestaña, eliminando cualquier aviso temporal.
  window.addEventListener("load",()=>{if(window.Plotly){$("aviso-cdn").style.display="none";refresh();}});
  window.addEventListener("resize",()=>{if(window.Plotly)document.querySelectorAll(".section.active .js-plotly-plot").forEach(el=>Plotly.Plots.resize(el));});
})();
</script>
</body>
</html>
'''


# -----------------------------------------------------------------------------
# 4. GENERACIÓN, COPIA PARA GITHUB PAGES Y VERIFICACIÓN
# -----------------------------------------------------------------------------
def generar_html(paquete: dict, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    html = PLANTILLA_HTML.replace("__DATOS_JSON__", escapar_json_script(paquete))
    if "__DATOS_JSON__" in html:
        raise RuntimeError("No se sustituyó la información incrustada en la plantilla.")
    ruta.write_text(html, encoding="utf-8")


def main() -> None:
    args = argumentos()
    df, auditoria = cargar_datos()
    paquete, ausentes = preparar_paquete(df, auditoria, args)
    generar_html(paquete, HTML_SALIDA)
    destinos = [HTML_SALIDA]
    if args.copiar in ("docs", "ambos"):
        salida_docs = RAIZ / "docs" / "index.html"
        salida_docs.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HTML_SALIDA, salida_docs)
        destinos.append(salida_docs)
    if args.copiar in ("raiz", "ambos"):
        salida_raiz = RAIZ / "index.html"
        shutil.copyfile(HTML_SALIDA, salida_raiz)
        destinos.append(salida_raiz)

    print("=" * 78)
    print("DASHBOARD SAE · GENERACIÓN COMPLETADA")
    print("=" * 78)
    print(f"Registros incrustados       : {len(df)}")
    print(f"Con flag_anomalia           : {int(df['flag_anomalia'].sum())}")
    print(f"Tablas opcionales integradas: {len(paquete['tablas'])} de {len(TABLAS_OPCIONALES)}")
    print(f"Resúmenes integrados        : {len(paquete['resumenes'])} de {len(RESUMENES)}")
    print(f"Fuentes opcionales ausentes : {len(ausentes)} (no impiden ejecutar)")
    print("Filtros: condición, familia, %C, propiedad, anomalías, valores originales/corregidos.")
    print("Gráficas: Plotly.js por CDN HTTPS; requiere internet al abrir el HTML.")
    print("\nArchivos HTML generados:")
    for ruta in destinos:
        print(f"  - {ruta.relative_to(RAIZ).as_posix()} ({ruta.stat().st_size / 1024:.1f} KiB)")
    print("\nPara GitHub Pages: usa --copiar docs y configura Pages desde /docs,")
    print("o usa --copiar raiz y configura Pages desde /(root).")
    print("=" * 78)


if __name__ == "__main__":
    main()
