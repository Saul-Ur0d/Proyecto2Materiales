#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_limpieza_datos.py
====================
Limpieza y auditoría de la base "Steel- Property and Composition (Carbon Steel)".

Entrada
-------
data/raw/Steel- Property and Composition (Carbon Steel).csv

Salidas
-------
data/processed/aceros_limpio.csv            -> dataset limpio (190 registros, sin borrar ninguno)
data/processed/auditoria_inconsistencias.csv -> un renglón por registro con hallazgos

Qué hace
--------
1. Convierte columnas numéricas con formato irregular (comas decimales, espacios, unidades).
2. Separa escalas de dureza (HB vs HRB) sin mezclarlas.
3. Normaliza la columna 'Conditions' (erratas OCR, saltos de línea, 'T' por '°F', mayúsculas)
   y la agrupa en categorías de procesamiento.
4. Calcula el %C representativo (C_pct) y clasifica la familia SAE.
5. Concilia las escalas MPa/ksi y detecta valores físicamente imposibles.
6. Audita TODO: ninguna fila se elimina; cada hallazgo queda registrado.

Convención de la auditoría
--------------------------
Cada hallazgo se escribe como  [TIPO:CATEGORIA] descripción, con TIPO en:
  - ANOMALIA   : dato sospechoso/imposible. Activa flag_anomalia = True.
  - CORRECCION : transformación automática aplicada (queda documentada).
  - AVISO      : limitación informativa (no es error, pero conviene saberla).

Uso
---
    python src/01_limpieza_datos.py

Nota: los CSV de salida se guardan con 'utf-8-sig' (para que Excel muestre bien '°').
Al leerlos en los siguientes scripts usar: pd.read_csv(ruta, encoding="utf-8-sig").
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# 1. RUTAS (relativas a la raíz del repositorio, sin importar desde dónde se ejecute)
# ----------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[1]
DIR_RAW = RAIZ / "data" / "raw"
DIR_PROCESSED = RAIZ / "data" / "processed"

ARCHIVO_RAW = DIR_RAW / "Steel- Property and Composition (Carbon Steel).csv"
ARCHIVO_LIMPIO = DIR_PROCESSED / "aceros_limpio.csv"
ARCHIVO_AUDITORIA = DIR_PROCESSED / "auditoria_inconsistencias.csv"

# ----------------------------------------------------------------------------
# 2. CONSTANTES
# ----------------------------------------------------------------------------
KSI_A_MPA = 6.89476          # 1 ksi = 6.89476 MPa
TOL_RATIO_KSI = 0.06         # tolerancia relativa (6 %) de la razón MPa/ksi (ksi viene redondeado)
TOL_TEMP_F = 10              # tolerancia (°F) en la conversión °C <-> °F de las condiciones
C_MAX_PLAUSIBLE = 1.5        # % C máximo plausible para estos aceros (límite pedido en la auditoría)

COLS_PROPIEDADES = [
    "UTS (MPa)", "UTS (Ksi)", "YS (MPa)", "YS (ksi)", "Elongation (%)", "Reduction (%)",
]
ELEMENTOS = ["C", "Mn", "P", "S", "Si", "Ni", "Cr", "Mo", "Ti"]
COLS_COMPOSICION = [f"{el} ({lim})" for el in ELEMENTOS for lim in ("Min", "Max")]
COLS_REQUERIDAS = COLS_PROPIEDADES + COLS_COMPOSICION + ["SAE Grade", "Conditions", "Hardness (HB)"]

# Columnas clave: si faltan, el registro es poco útil para el análisis
COLS_CLAVE = ["UTS (MPa)", "YS (MPa)", "Elongation (%)", "Reduction (%)", "Hardness_raw_num", "C_pct"]

# Erratas OCR evidentes en 'Conditions' (se aplican sobre el texto en minúsculas)
ERRATAS_CONDICIONES = [
    (re.compile(r"\bhot\s+rotted\b"), "hot rolled", "errata OCR 'rotted' -> 'rolled'"),
    (re.compile(r"\bcold\s+orawn\b"), "cold drawn", "errata OCR 'Orawn' -> 'drawn'"),
    (re.compile(r"\boi\s+i\s+quenched\b"), "oil quenched", "errata OCR 'Oi I' -> 'Oil'"),
]


# ----------------------------------------------------------------------------
# 3. REGISTRO DE HALLAZGOS (auditoría)
# ----------------------------------------------------------------------------
class RegistroHallazgos:
    """Acumula los hallazgos de auditoría por registro (índice del DataFrame)."""

    ORDEN_TIPOS = {"ANOMALIA": 0, "CORRECCION": 1, "AVISO": 2}

    def __init__(self, indice) -> None:
        self.datos: dict[int, list[tuple[str, str, str]]] = {i: [] for i in indice}

    def agregar(self, idx: int, tipo: str, categoria: str, texto: str) -> None:
        self.datos[idx].append((tipo, categoria, texto))

    def agregar_mascara(self, mascara: pd.Series, tipo: str, categoria: str, texto) -> None:
        """Agrega un hallazgo a cada fila donde la máscara es True.
        `texto` puede ser un str fijo o una función f(idx) -> str."""
        mascara = mascara.fillna(False).astype(bool)
        for idx in mascara.index[mascara]:
            self.agregar(idx, tipo, categoria, texto(idx) if callable(texto) else texto)

    def n_anomalias(self, idx: int) -> int:
        return sum(1 for t, _, _ in self.datos[idx] if t == "ANOMALIA")

    def motivo(self, idx: int) -> str:
        items = sorted(self.datos[idx], key=lambda x: self.ORDEN_TIPOS[x[0]])
        return " | ".join(f"[{t}:{c}] {txt}" for t, c, txt in items)

    def contar(self) -> Counter:
        """Cuenta cuántos registros tienen cada (tipo, categoría)."""
        cnt: Counter = Counter()
        for hallazgos in self.datos.values():
            for tipo, cat in {(t, c) for t, c, _ in hallazgos}:
                cnt[(tipo, cat)] += 1
        return cnt


def fmt(x) -> str:
    """Formato corto para mostrar números en los textos de auditoría."""
    return "NA" if pd.isna(x) else f"{x:g}"


# ----------------------------------------------------------------------------
# 4. LECTURA
# ----------------------------------------------------------------------------
def localizar_csv_crudo() -> Path:
    """Devuelve la ruta del CSV crudo. Si el nombre exacto no existe pero hay un
    único .csv en data/raw/, usa ese (útil si el archivo se renombró al subirlo)."""
    if ARCHIVO_RAW.exists():
        return ARCHIVO_RAW
    candidatos = sorted(DIR_RAW.glob("*.csv"))
    if len(candidatos) == 1:
        print(f"[AVISO] No se encontró '{ARCHIVO_RAW.name}'; se usará '{candidatos[0].name}'.")
        return candidatos[0]
    raise FileNotFoundError(
        f"No se encontró el CSV crudo en {DIR_RAW}. Se esperaba: '{ARCHIVO_RAW.name}'."
    )


def leer_crudo(ruta: Path) -> pd.DataFrame:
    """Lee el CSV como texto (dtype=str) para limpiar nosotros mismos cada columna."""
    df = None
    for enc in ("utf-8-sig", "latin-1"):  # latin-1 nunca falla: es el último recurso
        try:
            df = pd.read_csv(ruta, dtype=str, encoding=enc)
            break
        except UnicodeDecodeError:
            continue

    # Nombres de columnas homogéneos: sin espacios extra y 'S(Max)' -> 'S (Max)'
    df.columns = [re.sub(r"\s+", " ", re.sub(r"(\w)\(", r"\1 (", c.strip())) for c in df.columns]

    faltantes = [c for c in COLS_REQUERIDAS if c not in df.columns]
    if faltantes:
        raise KeyError(f"Faltan columnas esperadas en el CSV: {faltantes}")

    df = df.replace(r"^\s*$", np.nan, regex=True)  # celdas vacías/solo espacios -> NaN
    df["SAE Grade"] = df["SAE Grade"].str.strip().str.upper()
    df.insert(0, "registro", np.arange(1, len(df) + 1))  # nº de registro (1 = primera fila de datos)
    return df


# ----------------------------------------------------------------------------
# 5. LIMPIEZA NUMÉRICA
# ----------------------------------------------------------------------------
def a_numerico(serie: pd.Series) -> pd.Series:
    """Convierte texto a float: coma decimal -> punto, ignora espacios y unidades."""
    s = serie.astype("string").str.strip().str.replace(",", ".", regex=False)
    numero = s.str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(numero, errors="coerce").astype("float64")


def limpiar_columnas_numericas(df: pd.DataFrame, h: RegistroHallazgos) -> pd.DataFrame:
    """Convierte propiedades y composición a numérico, registrando cada corrección."""
    for col in COLS_PROPIEDADES + COLS_COMPOSICION:
        crudo = df[col].astype("string")
        limpio = a_numerico(crudo)

        h.agregar_mascara(
            crudo.str.contains(",", na=False), "CORRECCION", "COMA_DECIMAL",
            lambda i, c=col, cr=crudo: f"{c}: coma decimal '{cr[i]}' convertida a punto ({fmt(limpio[i])})",
        )
        h.agregar_mascara(
            crudo.str.contains(r"[A-Za-z%°]", na=False), "CORRECCION", "LIMPIEZA_NUMERICA",
            lambda i, c=col, cr=crudo: f"{c}: texto/unidad eliminado de '{cr[i]}'",
        )
        h.agregar_mascara(
            crudo.notna() & (crudo != crudo.str.strip()), "CORRECCION", "LIMPIEZA_NUMERICA",
            lambda i, c=col: f"{c}: espacios sobrantes eliminados",
        )
        h.agregar_mascara(
            crudo.notna() & limpio.isna(), "ANOMALIA", "NO_NUMERICO",
            lambda i, c=col, cr=crudo: f"{c}: valor '{cr[i]}' no se pudo convertir a número",
        )
        df[col] = limpio
    return df


# ----------------------------------------------------------------------------
# 6. DUREZA: separar escalas (HB vs HRB) sin mezclarlas
# ----------------------------------------------------------------------------
PATRON_DUREZA = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*([A-Za-z]*)\s*$")
ALIAS_BRINELL = {"", "HB", "HBW", "BHN"}


def procesar_dureza(df: pd.DataFrame, h: RegistroHallazgos) -> pd.DataFrame:
    """Crea Hardness_raw_num, Hardness_escala y Hardness_HB_num (solo Brinell válido)."""
    valores, escalas = [], []
    for v in df["Hardness (HB)"]:
        if pd.isna(v):
            valores.append(np.nan)
            escalas.append(np.nan)
            continue
        m = PATRON_DUREZA.match(str(v))
        if not m:
            valores.append(np.nan)
            escalas.append("DESCONOCIDA")
            continue
        valores.append(float(m.group(1).replace(",", ".")))
        esc = m.group(2).upper()
        escalas.append("HB" if esc in ALIAS_BRINELL else esc)

    df["Hardness_raw_num"] = valores
    df["Hardness_escala"] = escalas
    df["Hardness_HB_num"] = df["Hardness_raw_num"].where(df["Hardness_escala"] == "HB")
    df = df.drop(columns=["Hardness (HB)"])

    fuera_hb = df["Hardness_escala"].notna() & (df["Hardness_escala"] != "HB")

    def texto_hrb(i):
        esc, val = df.at[i, "Hardness_escala"], df.at[i, "Hardness_raw_num"]
        msg = (f"Dureza reportada en escala {esc} ({fmt(val)} {esc}) dentro de la columna 'Hardness (HB)': "
               f"no es comparable con Brinell; Hardness_HB_num queda vacío")
        if esc == "HRB" and pd.notna(val) and val > 100:
            msg += f". Además {fmt(val)} excede el máximo de la escala HRB (~100): probable error de captura"
        return msg

    h.agregar_mascara(fuera_hb, "ANOMALIA", "DUREZA_ESCALA_NO_HB", texto_hrb)

    # Rango físico razonable para HB en aceros (50-750)
    hb = df["Hardness_HB_num"]
    h.agregar_mascara(
        (hb < 50) | (hb > 750), "ANOMALIA", "RANGO_FISICO",
        lambda i: f"Hardness_HB_num={fmt(hb[i])} fuera del rango físico 50-750 HB",
    )
    return df


# ----------------------------------------------------------------------------
# 7. NORMALIZACIÓN DE 'Conditions'
# ----------------------------------------------------------------------------
def normalizar_condicion(original) -> tuple:
    """Devuelve (texto_limpio, lista_de_correcciones, pares_de_temperatura).

    pares_de_temperatura = [(°C, °F), ...] en el orden en que aparecen en el texto.
    """
    if pd.isna(original):
        return np.nan, [], []

    correcciones: list[str] = []
    base = str(original)

    if re.search(r"[\r\n]", base):
        correcciones.append("salto de línea eliminado")
    if re.search(r"[ \t]{2,}", base) or base != base.strip():
        correcciones.append("espacios extra eliminados")

    base_ws = re.sub(r"\s+", " ", base).strip()   # espacios y saltos de línea colapsados
    t = base_ws.lower()                           # se trabaja en minúsculas y al final se capitaliza

    # (a) Erratas OCR evidentes
    for patron, reemplazo, descripcion in ERRATAS_CONDICIONES:
        t, n = patron.subn(reemplazo, t)
        if n:
            correcciones.append(descripcion)

    # (b) Temperaturas en °C: '860°C' / '860 C' -> '860 °C'
    def _celsius(m: re.Match) -> str:
        num = m.group(1)
        if m.group(0) != f"{num} °c":
            correcciones.append(f"formato de temperatura °C normalizado ('{m.group(0)}' -> '{num} °C')")
        return f"{num} °C"

    t = re.sub(r"(\d{3,4})\s*[°º˚]?\s*c\b", _celsius, t)

    # (c) Temperaturas en °F entre paréntesis: '(1600 T)', "(1750 'T)", '(1600 )' -> '(1600 °F)'
    def _fahrenheit(m: re.Match) -> str:
        num = m.group(1)
        if m.group(0) != f"({num} °f)":
            correcciones.append(f"unidad °F corregida ('{m.group(0)}' -> '({num} °F)')")
        return f"({num} °F)"

    t = re.sub(r"\(\s*(\d{3,4})[^\d()]*\)", _fahrenheit, t)

    # (d) Capitalización tipo oración ('Cold Drawn' -> 'Cold drawn')
    limpio = t[0].upper() + t[1:]
    if limpio != base_ws and limpio.lower() == base_ws.lower():
        correcciones.append("capitalización unificada")

    pares = [(int(c), int(f)) for c, f in re.findall(r"(\d{3,4}) °C \((\d{3,4}) °F\)", limpio)]
    return limpio, correcciones, pares


def clasificar_condicion(texto) -> str:
    """Agrupa la condición limpia en categorías de procesamiento.

    Nota: 'Spheroidized annealed, cold drawn' se agrupa en 'Spheroidized annealed'
    (el texto completo se conserva en Conditions_limpia).
    """
    if pd.isna(texto):
        return "Other"
    s = texto.lower()
    estirado = "cold drawn" in s
    if "quenched" in s and "tempered" in s:
        return "Quenched & tempered"
    if "spheroidized" in s:
        return "Spheroidized annealed"
    if "annealed" in s and estirado:
        return "Annealed + Cold drawn"
    if "normalized" in s and estirado:
        return "Normalized + Cold drawn"
    if "annealed" in s:
        return "Annealed"
    if "normalized" in s:
        return "Normalized"
    if "hot rolled" in s:
        return "Hot rolled"
    if estirado:
        return "Cold drawn"
    return "Other"


def procesar_condiciones(df: pd.DataFrame, h: RegistroHallazgos) -> pd.DataFrame:
    """Crea Conditions_original, Conditions_limpia, Condicion_Grupo y temperaturas numéricas."""
    df["Conditions_original"] = df["Conditions"]
    limpias, listas_pares = [], []

    for idx, original in df["Conditions"].items():
        limpia, correcciones, pares = normalizar_condicion(original)
        limpias.append(limpia)
        listas_pares.append(pares)
        if correcciones:
            h.agregar(idx, "CORRECCION", "CONDICION_CORREGIDA",
                      "Conditions: " + "; ".join(correcciones))

    df["Conditions_limpia"] = limpias
    df["Condicion_Grupo"] = df["Conditions_limpia"].map(clasificar_condicion)
    df["_pares_temp"] = listas_pares

    # Temperaturas numéricas (útiles para el análisis de recocido/normalizado/temple)
    df["Temp_C"] = [p[0][0] if len(p) >= 1 else np.nan for p in listas_pares]
    df["Temp_revenido_C"] = [p[1][0] if len(p) >= 2 else np.nan for p in listas_pares]

    h.agregar_mascara(df["Condicion_Grupo"] == "Other", "AVISO", "CONDICION_SIN_CLASIFICAR",
                      "Condición no clasificada: agrupada como 'Other'")
    return df.drop(columns=["Conditions"])


# ----------------------------------------------------------------------------
# 8. CARBONO Y FAMILIA SAE
# ----------------------------------------------------------------------------
def clasificar_familia(grado) -> tuple[str, str]:
    """Devuelve (Familia_SAE, Serie_SAE). Ej.: '4140' -> ('Aleados 3xxx-9xxx', '41xx')."""
    m = re.match(r"^(\d{2})", str(grado))
    if not m:
        return "Otro", "Otro"
    prefijo = int(m.group(1))
    serie = f"{prefijo}xx"
    if prefijo == 10:
        return "Carbono simple 10xx", serie
    if prefijo in (11, 12):  # incluye 12L14
        return "Resulfurados/Refosforados 11xx-12xx", serie
    if prefijo in (13, 14, 15):
        return "Alto Mn 13xx-15xx", serie
    if 30 <= prefijo <= 99:
        return "Aleados 3xxx-9xxx", serie
    return "Otro", serie


def procesar_carbono_y_familia(df: pd.DataFrame, h: RegistroHallazgos) -> pd.DataFrame:
    """Calcula C_pct (promedio Min/Max o el valor disponible) y la familia SAE."""
    c_min, c_max = df["C (Min)"], df["C (Max)"]

    # Corrección documentada de C (Max) imposible (p. ej. 12L14: 15 -> 0.15)
    imposible = c_max > C_MAX_PLAUSIBLE
    corregible = imposible & ((c_max / 100) <= C_MAX_PLAUSIBLE)
    df["C_Max_corr"] = c_max.where(~corregible, c_max / 100)

    def texto_c(i):
        if corregible[i]:
            return (f"C (Max)={fmt(c_max[i])} % supera {C_MAX_PLAUSIBLE} %: imposible en estos aceros; "
                    f"probable punto decimal omitido. C_pct usa C_Max_corr={fmt(df.at[i, 'C_Max_corr'])} (valor/100)")
        return f"C (Max)={fmt(c_max[i])} % supera {C_MAX_PLAUSIBLE} % y no se pudo corregir automáticamente"

    h.agregar_mascara(imposible, "ANOMALIA", "C_MAX_IMPOSIBLE", texto_c)

    cmax_c = df["C_Max_corr"]
    ambos = c_min.notna() & cmax_c.notna()
    solo_max = c_min.isna() & cmax_c.notna()
    solo_min = c_min.notna() & cmax_c.isna()

    df["C_pct"] = np.select(
        [ambos, solo_max, solo_min],
        [(c_min + cmax_c) / 2, cmax_c, c_min],
        default=np.nan,
    ).round(4)
    df["C_pct_fuente"] = np.select(
        [ambos & ~corregible, ambos & corregible, solo_max & ~corregible, solo_max & corregible, solo_min],
        ["promedio Min-Max", "promedio Min-Max (Max corregido)", "solo C (Max)",
         "solo C (Max) corregido", "solo C (Min)"],
        default="sin dato",
    )

    h.agregar_mascara(
        df["C_pct_fuente"].str.startswith("solo C (Max)"), "AVISO", "C_SOLO_MAX",
        "C (Min) ausente: C_pct se estima solo con C (Max) (puede sobreestimar el %C real del rango)",
    )

    fam = df["SAE Grade"].map(clasificar_familia)
    df["Familia_SAE"] = fam.map(lambda x: x[0])
    df["Serie_SAE"] = fam.map(lambda x: x[1])
    return df


# ----------------------------------------------------------------------------
# 9. CONCILIACIÓN MPa / ksi Y PROPIEDADES IMPOSIBLES
# ----------------------------------------------------------------------------
def razon_mpa_ksi(mpa: pd.Series, ksi: pd.Series) -> pd.Series:
    """Razón MPa/ksi (debería ser ≈ 6.895). Devuelve NaN si falta algún dato."""
    return mpa / ksi.replace(0, np.nan)


def conciliar_escalas(df: pd.DataFrame, h: RegistroHallazgos) -> pd.DataFrame:
    """Detecta inconsistencias UTS/YS (MPa vs ksi, YS > UTS) y crea UTS_MPa_corr / YS_MPa_corr.

    Criterio (los valores originales NUNCA se sobrescriben):
      * MPa es la variable de referencia.
      * Si UTS (MPa) < YS (MPa) y el valor en ksi (convertido) sí es coherente con YS,
        se asume error en UTS (MPa) y se usa el valor convertido desde ksi.
      * Si YS (MPa) > UTS corregido y el valor en ksi (convertido) sí es coherente,
        se asume error en YS (MPa) y se usa el valor convertido desde ksi.
      * En cualquier otro caso se conserva el valor en MPa y se marca el ksi como erróneo.
    """
    uts, ys = df["UTS (MPa)"], df["YS (MPa)"]
    uts_desde_ksi = df["UTS (Ksi)"] * KSI_A_MPA
    ys_desde_ksi = df["YS (ksi)"] * KSI_A_MPA

    uts_corr, ys_corr = uts.copy(), ys.copy()

    cambia_uts = (uts < ys) & uts_desde_ksi.notna() & (uts_desde_ksi >= ys)
    uts_corr[cambia_uts] = uts_desde_ksi[cambia_uts].round()

    cambia_ys = (ys > uts_corr) & ys_desde_ksi.notna() & (ys_desde_ksi <= uts_corr)
    ys_corr[cambia_ys] = ys_desde_ksi[cambia_ys].round()

    df["UTS_MPa_corr"], df["YS_MPa_corr"] = uts_corr, ys_corr

    h.agregar_mascara(
        cambia_uts, "CORRECCION", "UTS_CORREGIDO",
        lambda i: (f"UTS_MPa_corr={fmt(uts_corr[i])} reemplaza UTS (MPa)={fmt(uts[i])} "
                   f"(derivado de UTS (Ksi)={fmt(df.at[i, 'UTS (Ksi)'])} × {KSI_A_MPA})"),
    )
    h.agregar_mascara(
        cambia_ys, "CORRECCION", "YS_CORREGIDO",
        lambda i: (f"YS_MPa_corr={fmt(ys_corr[i])} reemplaza YS (MPa)={fmt(ys[i])} "
                   f"(derivado de YS (ksi)={fmt(df.at[i, 'YS (ksi)'])} × {KSI_A_MPA})"),
    )

    # YS > UTS en los datos originales
    h.agregar_mascara(
        ys > uts, "ANOMALIA", "YS_MAYOR_UTS",
        lambda i: f"YS (MPa)={fmt(ys[i])} > UTS (MPa)={fmt(uts[i])}: físicamente imposible (YS debe ser ≤ UTS)",
    )

    # Razón MPa/ksi fuera de lo esperado (errores de escala/OCR en las columnas ksi o MPa)
    for nombre, mpa, ksi, cambia in (
        ("UTS", uts, df["UTS (Ksi)"], cambia_uts),
        ("YS", ys, df["YS (ksi)"], cambia_ys),
    ):
        razon = razon_mpa_ksi(mpa, ksi)
        desviada = razon.notna() & ((razon / KSI_A_MPA - 1).abs() > TOL_RATIO_KSI)

        def texto_razon(i, n=nombre, m=mpa, k=ksi, r=razon, c=cambia):
            base = (f"{n} (MPa)={fmt(m[i])} vs {n} (ksi)={fmt(k[i])}: razón MPa/ksi={r[i]:.2f} "
                    f"(esperado ≈{KSI_A_MPA:.2f} ±{TOL_RATIO_KSI:.0%}); ")
            if c[i]:
                return base + f"se asume error en {n} (MPa) y se corrige con el valor en ksi"
            return base + f"se asume error de captura en {n} (ksi); se conserva {n} (MPa)"

        h.agregar_mascara(desviada, "ANOMALIA", f"ESCALA_{nombre}_MPA_KSI", texto_razon)

    # Valor en ksi faltante: es un aviso (la columna MPa existe y es la de referencia)
    h.agregar_mascara(df["UTS (Ksi)"].isna(), "AVISO", "KSI_FALTANTE",
                      "UTS (Ksi) faltante (se conserva UTS (MPa))")
    h.agregar_mascara(df["YS (ksi)"].isna(), "AVISO", "KSI_FALTANTE",
                      "YS (ksi) faltante (se conserva YS (MPa))")
    return df


# ----------------------------------------------------------------------------
# 10. VALIDACIONES GENERALES
# ----------------------------------------------------------------------------
def validaciones_generales(df: pd.DataFrame, h: RegistroHallazgos) -> None:
    """Faltantes clave, rangos físicos, min>max, temperaturas y duplicados."""
    # Faltantes clave
    for col in COLS_CLAVE:
        h.agregar_mascara(df[col].isna(), "ANOMALIA", "FALTANTE_CLAVE",
                          f"Valor clave faltante: {col}")

    # Porcentajes fuera de 0-100
    for col in ("Elongation (%)", "Reduction (%)"):
        s = df[col]
        h.agregar_mascara((s < 0) | (s > 100), "ANOMALIA", "RANGO_FISICO",
                          lambda i, c=col, s=s: f"{c}={fmt(s[i])} fuera del rango 0-100 %")

    # Límites de composición invertidos (Min > Max)
    for el in ELEMENTOS:
        lo, hi = df[f"{el} (Min)"], df[f"{el} (Max)"]
        h.agregar_mascara(lo > hi, "ANOMALIA", "MIN_MAYOR_MAX",
                          lambda i, e=el, lo=lo, hi=hi: f"{e} (Min)={fmt(lo[i])} > {e} (Max)={fmt(hi[i])}")

    # Temperaturas: °C y °F deben corresponder entre sí
    def texto_temp(i):
        malos = []
        for c, f in df.at[i, "_pares_temp"]:
            esperado = c * 9 / 5 + 32
            if abs(esperado - f) > TOL_TEMP_F:
                malos.append(f"{c} °C ≈ {esperado:.0f} °F pero se registra {f} °F")
        return "Temperatura inconsistente en Conditions: " + "; ".join(malos)

    inconsistente = df["_pares_temp"].map(
        lambda ps: any(abs(c * 9 / 5 + 32 - f) > TOL_TEMP_F for c, f in ps)
    )
    h.agregar_mascara(inconsistente, "ANOMALIA", "TEMP_INCONSISTENTE", texto_temp)

    # Duplicados tras normalizar la condición
    duplicado = df.duplicated(["SAE Grade", "Conditions_limpia"], keep=False)
    h.agregar_mascara(duplicado, "ANOMALIA", "DUPLICADO",
                      "Registro duplicado (mismo SAE Grade y misma condición tras normalizar)")


# ----------------------------------------------------------------------------
# 11. EXPORTACIÓN
# ----------------------------------------------------------------------------
ORDEN_COLUMNAS = [
    "registro", "SAE Grade", "Familia_SAE", "Serie_SAE",
    "Conditions_original", "Conditions_limpia", "Condicion_Grupo", "Temp_C", "Temp_revenido_C",
    "UTS (MPa)", "UTS (Ksi)", "UTS_MPa_corr", "YS (MPa)", "YS (ksi)", "YS_MPa_corr",
    "Elongation (%)", "Reduction (%)",
    "Hardness_raw_num", "Hardness_escala", "Hardness_HB_num",
    "C_pct", "C_pct_fuente", "C (Min)", "C (Max)", "C_Max_corr",
] + [c for c in COLS_COMPOSICION if not c.startswith("C ")] + [
    "n_anomalias", "flag_anomalia", "motivo_auditoria",
]

COLS_AUDITORIA = [
    "registro", "SAE Grade", "Familia_SAE", "Conditions_original", "Conditions_limpia", "Condicion_Grupo",
    "UTS (MPa)", "UTS (Ksi)", "UTS_MPa_corr", "YS (MPa)", "YS (ksi)", "YS_MPa_corr",
    "Elongation (%)", "Reduction (%)", "Hardness_raw_num", "Hardness_escala",
    "C (Min)", "C (Max)", "C_Max_corr", "C_pct",
    "n_anomalias", "flag_anomalia", "motivo_auditoria",
]


def exportar(df: pd.DataFrame, h: RegistroHallazgos) -> pd.DataFrame:
    """Agrega columnas de auditoría, guarda ambos CSV y devuelve la tabla de auditoría."""
    df["n_anomalias"] = [h.n_anomalias(i) for i in df.index]
    df["flag_anomalia"] = df["n_anomalias"] > 0
    df["motivo_auditoria"] = [h.motivo(i) for i in df.index]

    DIR_PROCESSED.mkdir(parents=True, exist_ok=True)

    limpio = df[ORDEN_COLUMNAS]
    limpio.to_csv(ARCHIVO_LIMPIO, index=False, encoding="utf-8-sig")

    auditoria = (
        df.loc[df["motivo_auditoria"] != "", COLS_AUDITORIA]
        .sort_values(["flag_anomalia", "registro"], ascending=[False, True])
    )
    auditoria.to_csv(ARCHIVO_AUDITORIA, index=False, encoding="utf-8-sig")
    return auditoria


def imprimir_resumen(df: pd.DataFrame, auditoria: pd.DataFrame, h: RegistroHallazgos,
                     n_filas_raw: int, n_cols_raw: int, ruta_raw: Path) -> None:
    """Resumen legible en consola."""
    linea = "=" * 78
    print(linea)
    print("RESUMEN DE LIMPIEZA Y AUDITORÍA")
    print(linea)
    print(f"Archivo de entrada : {ruta_raw.relative_to(RAIZ)}")
    print(f"Filas procesadas   : {n_filas_raw} (columnas originales: {n_cols_raw}); filas eliminadas: 0")
    print(f"Filas con anomalía : {int(df['flag_anomalia'].sum())}")
    print(f"Filas con hallazgos (anomalía, corrección o aviso): {len(auditoria)}")

    print("\nHallazgos por tipo y categoría (nº de registros afectados):")
    conteo = h.contar()
    for (tipo, cat), n in sorted(conteo.items(), key=lambda x: (h.ORDEN_TIPOS[x[0][0]], -x[1], x[0][1])):
        print(f"  {tipo:<11} {cat:<28} {n:>4}")

    print("\nRegistros con ANOMALIA:")
    anomalos = df[df["flag_anomalia"]]
    for _, fila in anomalos.iterrows():
        cats = sorted({c for t, c, _ in h.datos[fila.name] if t == "ANOMALIA"})
        cond = str(fila["Conditions_limpia"])
        cond = cond if len(cond) <= 38 else cond[:35] + "..."
        print(f"  #{int(fila['registro']):>3} SAE {fila['SAE Grade']:<6} {cond:<38} -> {', '.join(cats)}")

    print("\nDistribución por Condicion_Grupo:")
    print(df["Condicion_Grupo"].value_counts().to_string())
    print("\nDistribución por Familia_SAE:")
    print(df["Familia_SAE"].value_counts().to_string())
    print("\nEscalas de dureza:")
    print(df["Hardness_escala"].value_counts(dropna=False).to_string())

    print("\nArchivos generados:")
    print(f"  - {ARCHIVO_LIMPIO.relative_to(RAIZ)}  ({len(df)} filas x {len(ORDEN_COLUMNAS)} columnas)")
    print(f"  - {ARCHIVO_AUDITORIA.relative_to(RAIZ)}  ({len(auditoria)} filas)")
    print(linea)


# ----------------------------------------------------------------------------
# 12. PROGRAMA PRINCIPAL
# ----------------------------------------------------------------------------
def main() -> None:
    ruta_raw = localizar_csv_crudo()
    df = leer_crudo(ruta_raw)
    n_filas_raw, n_cols_raw = df.shape[0], df.shape[1] - 1  # -1: columna 'registro' agregada

    h = RegistroHallazgos(df.index)

    df = limpiar_columnas_numericas(df, h)   # comas decimales, espacios, unidades
    df = procesar_dureza(df, h)              # HB vs HRB
    df = procesar_condiciones(df, h)         # erratas OCR + grupos de proceso
    df = procesar_carbono_y_familia(df, h)   # C_pct, Familia_SAE
    df = conciliar_escalas(df, h)            # MPa vs ksi, YS > UTS
    validaciones_generales(df, h)            # faltantes, rangos, temperaturas, duplicados

    auditoria = exportar(df, h)
    imprimir_resumen(df, auditoria, h, n_filas_raw, n_cols_raw, ruta_raw)


if __name__ == "__main__":
    main()