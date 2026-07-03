"""
==========================================================================
  Mundiales 2018, 2022 y 2026 — Fase de Grupos
  Preparación de datos y entrada al análisis supervisado
  Script completo: Perfilado → Limpieza → Comparación → Clasificación
==========================================================================
"""

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

# ── Configuración ──
DATA = Path(__file__).resolve().parent / "datos"
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  PARTE A — CARGA Y PERFILADO                                       ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("=" * 70)
print("  PARTE A — CARGA Y PERFILADO DE DATOS")
print("=" * 70)

d18 = pd.read_csv(DATA / "mundial_2018_sucio.csv", dtype=str)
d22 = pd.read_csv(DATA / "mundial_2022_sucio.csv", dtype=str)
d26 = pd.read_csv(DATA / "mundial_2026_sucio.csv", dtype=str)


def perfil(df, nombre):
    """Muestra un perfil completo de calidad de un DataFrame."""
    print(f"\n{'─' * 60}")
    print(f"  PERFIL: Mundial {nombre}")
    print(f"{'─' * 60}")
    print(f"  Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
    print(f"  Columnas: {list(df.columns)}")
    print(f"\n  Valores nulos por columna:")
    nulos = df.isnull().sum()
    for col in df.columns:
        if nulos[col] > 0:
            print(f"    → {col}: {nulos[col]}")
    if nulos.sum() == 0:
        print(f"    (ninguno)")

    # Duplicados por ID
    id_col = df.columns[0]
    dupes = df[id_col].duplicated(keep=False)
    n_dupes = dupes.sum()
    if n_dupes > 0:
        ids_dup = df.loc[dupes, id_col].unique()
        print(f"\n  ⚠ Duplicados por '{id_col}': {n_dupes} filas → IDs: {list(ids_dup)}")
    else:
        print(f"\n  ✓ Sin duplicados por '{id_col}'")

    # Equipos únicos
    cols_equipo = [c for c in df.columns if any(k in c.lower() for k in ["local", "home", "visit", "away"])]
    equipos_unicos = set()
    for c in cols_equipo:
        equipos_unicos.update(df[c].dropna().str.strip().unique())
    print(f"\n  Equipos únicos encontrados ({len(equipos_unicos)}):")
    for eq in sorted(equipos_unicos):
        print(f"    • {repr(eq)}")

    # Formatos de grupo
    cols_grupo = [c for c in df.columns if any(k in c.lower() for k in ["grupo", "group", "grp"])]
    if cols_grupo:
        grupos = df[cols_grupo[0]].dropna().unique()
        print(f"\n  Valores únicos de grupo: {list(grupos)}")

    # Formatos de fase
    cols_fase = [c for c in df.columns if any(k in c.lower() for k in ["fase", "stage", "round"])]
    if cols_fase:
        fases = df[cols_fase[0]].dropna().unique()
        print(f"  Valores únicos de fase: {list(fases)}")

    # Goles problemáticos
    cols_goles = [c for c in df.columns if any(k in c.lower() for k in ["gol", "score", "hg", "ag"])]
    for c in cols_goles:
        problemas = []
        for v in df[c].dropna().unique():
            v_stripped = v.strip()
            try:
                num = int(v_stripped)
                if num < 0:
                    problemas.append(f"{repr(v)} (negativo)")
            except ValueError:
                problemas.append(repr(v))
        if problemas:
            print(f"\n  ⚠ Goles problemáticos en '{c}': {problemas}")

    print()


perfil(d18, "2018")
perfil(d22, "2022")
perfil(d26, "2026")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  PARTE B — LIMPIEZA Y UNIFICACIÓN                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 70)
print("  PARTE B — LIMPIEZA Y UNIFICACIÓN")
print("=" * 70)

# ── 1. Catálogo de equipos ──

catalogo = pd.read_csv(DATA / "catalogo_equipos.csv")


def clave_texto(valor):
    """Normaliza un texto a clave de búsqueda: sin acentos, minúsculas,
    solo letras y espacios."""
    if pd.isna(valor):
        return ""
    valor = str(valor).strip()
    # Descomponer caracteres Unicode y quitar marcas diacríticas
    nfkd = unicodedata.normalize("NFKD", valor)
    sin_acentos = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Minúsculas, quitar todo excepto letras y espacios, colapsar espacios
    limpio = re.sub(r"[^a-zA-Z\s]", " ", sin_acentos).lower()
    limpio = re.sub(r"\s+", " ", limpio).strip()
    return limpio


# Construir diccionario clave → nombre_canónico
dict_equipos = {}
for _, fila in catalogo.iterrows():
    clave = clave_texto(fila["variante"])
    dict_equipos[clave] = fila["nombre_canonico"]

# Verificar: debe poder mapear todas las variantes
print(f"\n  Catálogo cargado: {len(dict_equipos)} variantes → {len(set(dict_equipos.values()))} equipos canónicos")


def normalizar_equipo(valor):
    """Convierte un nombre de equipo a su forma canónica usando el catálogo."""
    clave = clave_texto(valor)
    if clave in dict_equipos:
        return dict_equipos[clave]
    # Intentar sin 'the' al inicio
    sin_the = re.sub(r"^the\s+", "", clave)
    if sin_the in dict_equipos:
        return dict_equipos[sin_the]
    print(f"    ⚠ Equipo no encontrado en catálogo: {repr(valor)} → clave: {repr(clave)}")
    return valor.strip()


# ── 2. Funciones de limpieza ──

# Rangos de fechas válidos para cada mundial
RANGOS_FECHAS = {
    2018: (pd.Timestamp("2018-06-14"), pd.Timestamp("2018-06-28")),
    2022: (pd.Timestamp("2022-11-20"), pd.Timestamp("2022-12-02")),
    2026: (pd.Timestamp("2026-06-11"), pd.Timestamp("2026-06-27")),
}

# Origen de fechas seriales de Excel (sistema 1900, Windows)
EXCEL_EPOCH = pd.Timestamp("1899-12-30")


def convertir_fecha(valor, mundial):
    """Convierte un valor de fecha en múltiples formatos posibles a datetime."""
    if pd.isna(valor) or str(valor).strip().upper() in ("", "N/D", "S/D"):
        return pd.NaT

    valor = str(valor).strip()
    rango_min, rango_max = RANGOS_FECHAS[mundial]

    # 1. ¿Es un serial de Excel? (número entero grande, e.g. 46187)
    try:
        serial = int(valor)
        if 40000 < serial < 50000:
            fecha = EXCEL_EPOCH + pd.Timedelta(days=serial)
            if rango_min <= fecha <= rango_max:
                return fecha
    except (ValueError, TypeError):
        pass

    # 2. Intentar múltiples formatos de fecha
    formatos = [
        "%Y-%m-%d",       # 2018-06-14
        "%d/%m/%Y",       # 15/06/2018
        "%m/%d/%Y",       # 06/15/2018
        "%d-%m-%y",       # 15-06-18
        "%b %d, %Y",      # Jun 16, 2018
    ]

    candidatas = []
    for fmt in formatos:
        try:
            fecha = pd.to_datetime(valor, format=fmt)
            candidatas.append(fecha)
        except (ValueError, TypeError):
            continue

    # Filtrar las que caen dentro del rango válido
    validas = [f for f in candidatas if rango_min <= f <= rango_max]
    if validas:
        return validas[0]

    # Si ninguna cae en el rango, devolver la primera candidata
    if candidatas:
        return candidatas[0]

    # Último intento: parseo libre de pandas
    try:
        fecha = pd.to_datetime(valor, dayfirst=True)
        return fecha
    except Exception:
        return pd.NaT


def extraer_numero(valor):
    """Extrae el primer entero de un valor de goles, o NaN si no es posible."""
    if pd.isna(valor):
        return np.nan
    valor = str(valor).strip()
    if valor.upper() in ("", "N/A", "S/D", "SIN DATO"):
        return np.nan
    # Quitar texto como "goles"
    valor = re.sub(r"[^\d\-]", "", valor)
    if not valor:
        return np.nan
    try:
        return int(valor)
    except (ValueError, TypeError):
        return np.nan


def separar_marcador(valor):
    """Separa un marcador como '2-1', '2 : 1', '2 x 1', '2–1', '2—1'
    en (goles_local, goles_visitante)."""
    if pd.isna(valor):
        return np.nan, np.nan
    valor = str(valor).strip()
    if valor.upper() in ("", "N/A", "S/D", "SIN DATO"):
        return np.nan, np.nan

    # Normalizar separadores: –, —, :, x, X → -
    normalizado = re.sub(r"\s*[–—:\xb7]\s*", "-", valor)
    normalizado = re.sub(r"\s+[xX]\s+", "-", normalizado)
    normalizado = normalizado.strip()

    partes = normalizado.split("-")
    if len(partes) == 2:
        try:
            g1 = int(partes[0].strip())
            g2 = int(partes[1].strip())
            return g1, g2
        except (ValueError, TypeError):
            return np.nan, np.nan

    return np.nan, np.nan


def normalizar_grupo(valor):
    """Extrae la letra del grupo (A-L) de cualquier formato."""
    if pd.isna(valor):
        return np.nan
    valor = str(valor).strip().upper()
    if valor in ("", "S/D"):
        return np.nan
    # Buscar la última letra A-L en el valor
    letras = re.findall(r"[A-L]", valor)
    if letras:
        return letras[-1]
    return np.nan


def normalizar_booleano(valor):
    """Convierte Sí, 1, TRUE, si, etc. a booleano."""
    if pd.isna(valor):
        return False
    valor = str(valor).strip().upper()
    return valor in ("SÍ", "SI", "1", "TRUE", "YES", "S")


# ── 3. Mapas de renombrado de columnas ──

rename_maps = {
    2018: {
        "ID Partido": "partido_id",
        "Año": "mundial",
        "Fase": "fase",
        "Grupo": "grupo",
        "Jornada": "jornada",
        "Fecha": "fecha",
        "Equipo Local": "equipo_local",
        "Equipo Visitante": "equipo_visitante",
        "Goles Local": "goles_local",
        "Goles Visitante": "goles_visitante",
        "Marcador": "marcador",
        "Anfitrión Local": "local_es_anfitrion",
        "Fuente": "fuente",
    },
    2022: {
        "match_id": "partido_id",
        "WorldCup": "mundial",
        "stage": "fase",
        "group_name": "grupo",
        "match_day": "jornada",
        "date": "fecha",
        "local": "equipo_local",
        "visitor": "equipo_visitante",
        "home_score": "goles_local",
        "away_score": "goles_visitante",
        "score_text": "marcador",
        "home_host": "local_es_anfitrion",
        "source_url": "fuente",
    },
    2026: {
        "match": "partido_id",
        "wc": "mundial",
        "round": "fase",
        "grp": "grupo",
        "md": "jornada",
        "played_on": "fecha",
        "home": "equipo_local",
        "away": "equipo_visitante",
        "HG": "goles_local",
        "AG": "goles_visitante",
        "result_raw": "marcador",
        "host_h": "local_es_anfitrion",
        "host_a": "visitante_es_anfitrion",
        "source": "fuente",
    },
}

# Esquema canónico
COLUMNAS_BASE = [
    "partido_id", "mundial", "fase", "grupo", "jornada", "fecha",
    "equipo_local", "equipo_visitante", "goles_local",
    "goles_visitante", "marcador", "local_es_anfitrion",
    "visitante_es_anfitrion", "fuente",
]


# ── 4. Función de limpieza principal ──

# Diccionario de grupo por equipo (para inferir grupos faltantes)
GRUPOS_2018 = {
    "Russia": "A", "Saudi Arabia": "A", "Egypt": "A", "Uruguay": "A",
    "Portugal": "B", "Spain": "B", "Morocco": "B", "Iran": "B",
    "France": "C", "Australia": "C", "Peru": "C", "Denmark": "C",
    "Argentina": "D", "Iceland": "D", "Croatia": "D", "Nigeria": "D",
    "Brazil": "E", "Switzerland": "E", "Costa Rica": "E", "Serbia": "E",
    "Germany": "F", "Mexico": "F", "Sweden": "F", "South Korea": "F",
    "Belgium": "G", "Panama": "G", "Tunisia": "G", "England": "G",
    "Poland": "H", "Senegal": "H", "Colombia": "H", "Japan": "H",
}


def limpiar_mundial(df, mundial):
    """Limpia y normaliza un DataFrame de un mundial específico."""
    df = df.copy()

    # 1. Renombrar columnas
    mapa = rename_maps[mundial]
    df = df.rename(columns=mapa)

    # Crear columnas faltantes
    if "visitante_es_anfitrion" not in df.columns:
        df["visitante_es_anfitrion"] = "0"
    if "fuente" not in df.columns:
        df["fuente"] = ""

    # Seleccionar solo columnas del esquema canónico (las que existan)
    cols_disponibles = [c for c in COLUMNAS_BASE if c in df.columns]
    df = df[cols_disponibles]

    # Añadir columnas que falten
    for col in COLUMNAS_BASE:
        if col not in df.columns:
            df[col] = np.nan

    # 2. Mundial como entero
    df["mundial"] = mundial

    # 3. Fase: siempre "Fase de grupos"
    df["fase"] = "Fase de grupos"

    # 4. Normalizar equipos
    df["equipo_local"] = df["equipo_local"].apply(normalizar_equipo)
    df["equipo_visitante"] = df["equipo_visitante"].apply(normalizar_equipo)

    # 5. Normalizar grupo
    df["grupo"] = df["grupo"].apply(normalizar_grupo)

    # 6. Inferir grupos faltantes usando los equipos
    if mundial == 2018:
        grupos_ref = GRUPOS_2018
    else:
        # Construir referencia del propio dataset (los que sí tienen grupo)
        grupos_ref = {}
        for _, fila in df.dropna(subset=["grupo"]).iterrows():
            grupos_ref[fila["equipo_local"]] = fila["grupo"]
            grupos_ref[fila["equipo_visitante"]] = fila["grupo"]

    mascara_sin_grupo = df["grupo"].isna()
    for idx in df[mascara_sin_grupo].index:
        local = df.at[idx, "equipo_local"]
        visitante = df.at[idx, "equipo_visitante"]
        if local in grupos_ref:
            df.at[idx, "grupo"] = grupos_ref[local]
        elif visitante in grupos_ref:
            df.at[idx, "grupo"] = grupos_ref[visitante]

    # 7. Convertir fechas
    df["fecha"] = df["fecha"].apply(lambda v: convertir_fecha(v, mundial))

    # 8. Jornada como entero
    df["jornada"] = df["jornada"].apply(extraer_numero).astype("Int64")

    # 9. Booleanos de anfitrión
    df["local_es_anfitrion"] = df["local_es_anfitrion"].apply(normalizar_booleano)
    df["visitante_es_anfitrion"] = df["visitante_es_anfitrion"].apply(normalizar_booleano)

    # 10. Separar marcador
    marcador_parsed = df["marcador"].apply(separar_marcador)
    df["goles_marcador_local"] = marcador_parsed.apply(lambda x: x[0])
    df["goles_marcador_visitante"] = marcador_parsed.apply(lambda x: x[1])

    # 11. Extraer goles de las columnas originales
    df["goles_local_raw"] = df["goles_local"].apply(extraer_numero)
    df["goles_visitante_raw"] = df["goles_visitante"].apply(extraer_numero)

    # 12. Reparar goles: priorizar marcador sobre columna individual
    # Regla: si el marcador es parseable, usar esos valores.
    # Si el marcador es vacío/inválido, usar los goles individuales.
    # Rechazar goles negativos siempre.
    def reparar_gol(raw, marcador_val):
        """Decide el valor final de un gol."""
        # Si el marcador es válido, usarlo
        if not np.isnan(marcador_val) and marcador_val >= 0:
            return int(marcador_val)
        # Si el raw es válido y no negativo, usarlo
        if not np.isnan(raw) and raw >= 0:
            return int(raw)
        return np.nan

    df["goles_local"] = df.apply(
        lambda r: reparar_gol(r["goles_local_raw"], r["goles_marcador_local"]), axis=1
    )
    df["goles_visitante"] = df.apply(
        lambda r: reparar_gol(r["goles_visitante_raw"], r["goles_marcador_visitante"]), axis=1
    )

    # Reconstruir marcador limpio
    df["marcador"] = df.apply(
        lambda r: f"{int(r['goles_local'])}-{int(r['goles_visitante'])}"
        if not (pd.isna(r["goles_local"]) or pd.isna(r["goles_visitante"]))
        else np.nan,
        axis=1,
    )

    # Limpiar columnas temporales
    df = df.drop(columns=[
        "goles_marcador_local", "goles_marcador_visitante",
        "goles_local_raw", "goles_visitante_raw",
    ])

    # 13. Eliminar duplicados por partido_id (conservar primera ocurrencia)
    n_antes = len(df)
    df = df.drop_duplicates(subset=["partido_id"], keep="first")
    n_despues = len(df)
    if n_antes != n_despues:
        print(f"  ✓ Duplicados eliminados en {mundial}: {n_antes - n_despues} filas")

    # 14. Crear columnas derivadas
    df["goles_local"] = df["goles_local"].astype("Int64")
    df["goles_visitante"] = df["goles_visitante"].astype("Int64")
    df["goles_totales"] = df["goles_local"] + df["goles_visitante"]
    df["diferencia_goles"] = df["goles_local"] - df["goles_visitante"]

    # Resultado del local
    def resultado_local(row):
        if pd.isna(row["goles_local"]) or pd.isna(row["goles_visitante"]):
            return np.nan
        if row["goles_local"] > row["goles_visitante"]:
            return "Gana"
        elif row["goles_local"] == row["goles_visitante"]:
            return "Empata"
        else:
            return "Pierde"

    df["resultado_local"] = df.apply(resultado_local, axis=1)

    # Puntos del local (3 victoria, 1 empate, 0 derrota)
    puntos_map = {"Gana": 3, "Empata": 1, "Pierde": 0}
    df["puntos_local"] = df["resultado_local"].map(puntos_map).astype("Int64")

    # Ordenar por fecha y jornada
    df = df.sort_values(["jornada", "fecha", "partido_id"]).reset_index(drop=True)

    return df


# ── 5. Ejecutar limpieza ──

print("\n  Limpiando 2018...")
limpio18 = limpiar_mundial(d18, 2018)
print(f"  → {len(limpio18)} partidos")

print("\n  Limpiando 2022...")
limpio22 = limpiar_mundial(d22, 2022)
print(f"  → {len(limpio22)} partidos")

print("\n  Limpiando 2026...")
limpio26 = limpiar_mundial(d26, 2026)
print(f"  → {len(limpio26)} partidos")

# Concatenar
partidos = pd.concat([limpio18, limpio22, limpio26], ignore_index=True)
print(f"\n  ✓ Base unificada: {len(partidos)} partidos totales")

# ── 6. Validaciones obligatorias ──

print("\n" + "─" * 60)
print("  VALIDACIONES")
print("─" * 60)

# Conteo por mundial
conteo = partidos.groupby("mundial").size()
assert conteo[2018] == 48, f"2018: esperados 48, obtenidos {conteo[2018]}"
assert conteo[2022] == 48, f"2022: esperados 48, obtenidos {conteo[2022]}"
assert conteo[2026] == 72, f"2026: esperados 72, obtenidos {conteo[2026]}"
print(f"  ✓ Partidos por mundial: 2018={conteo[2018]}, 2022={conteo[2022]}, 2026={conteo[2026]}")

# Total
assert len(partidos) == 168, f"Total: esperados 168, obtenidos {len(partidos)}"
print(f"  ✓ Total: {len(partidos)} partidos")

# Duplicados
dupes = partidos.duplicated(subset=["partido_id"]).sum()
assert dupes == 0, f"Hay {dupes} duplicados"
print(f"  ✓ Duplicados: {dupes}")

# Goles negativos
negativos_local = (partidos["goles_local"].dropna() < 0).sum()
negativos_visita = (partidos["goles_visitante"].dropna() < 0).sum()
assert negativos_local == 0, f"Hay {negativos_local} goles locales negativos"
assert negativos_visita == 0, f"Hay {negativos_visita} goles visitantes negativos"
print(f"  ✓ Goles negativos: 0")

# Nulos críticos
for col in ["equipo_local", "equipo_visitante", "goles_local", "goles_visitante", "grupo"]:
    n_nulos = partidos[col].isna().sum()
    assert n_nulos == 0, f"Hay {n_nulos} nulos en '{col}'"
print(f"  ✓ Sin nulos en columnas críticas (equipos, goles, grupo)")

# Consistencia marcador con goles
for _, r in partidos.iterrows():
    if pd.notna(r["marcador"]):
        esperado = f"{r['goles_local']}-{r['goles_visitante']}"
        assert r["marcador"] == esperado, \
            f"Marcador inconsistente en {r['partido_id']}: {r['marcador']} vs {esperado}"
print(f"  ✓ Marcador consistente con goles en todos los partidos")

# Grupos válidos
grupos_validos = set("ABCDEFGHIJKL")
grupos_usados = set(partidos["grupo"].dropna().unique())
assert grupos_usados.issubset(grupos_validos), f"Grupos inválidos: {grupos_usados - grupos_validos}"
print(f"  ✓ Todos los grupos son letras A-L")

print(f"\n  ══ TODAS LAS VALIDACIONES PASARON ══\n")

# Guardar base limpia
partidos.to_csv(DATA / "base_limpia_168.csv", index=False, encoding="utf-8-sig")
print(f"  ✓ Base limpia guardada en: datos/base_limpia_168.csv")

# Mostrar primeras filas
print("\n  Muestra de la base limpia:")
print(partidos[["partido_id", "mundial", "grupo", "jornada", "fecha",
                "equipo_local", "equipo_visitante", "marcador",
                "resultado_local"]].head(10).to_string(index=False))


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  PARTE C — COMPARACIÓN DE TORNEOS                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\n\n" + "=" * 70)
print("  PARTE C — COMPARACIÓN DE TORNEOS")
print("=" * 70)

# ── Tabla comparativa ──

comparacion = partidos.groupby("mundial").agg(
    partidos_n=("partido_id", "count"),
    goles_totales=("goles_totales", "sum"),
    goles_por_partido=("goles_totales", "mean"),
    empates=("resultado_local", lambda x: (x == "Empata").sum()),
    victorias_local=("resultado_local", lambda x: (x == "Gana").sum()),
    victorias_visitante=("resultado_local", lambda x: (x == "Pierde").sum()),
).reset_index()

comparacion["pct_empates"] = (comparacion["empates"] / comparacion["partidos_n"] * 100).round(1)
comparacion["pct_victorias_local"] = (comparacion["victorias_local"] / comparacion["partidos_n"] * 100).round(1)
comparacion["goles_por_partido"] = comparacion["goles_por_partido"].round(2)

# Proporción de partidos con más de 2.5 goles
for wc in [2018, 2022, 2026]:
    subset = partidos[partidos["mundial"] == wc]
    pct_over = (subset["goles_totales"] > 2).mean() * 100
    comparacion.loc[comparacion["mundial"] == wc, "pct_over_2_5_goles"] = round(pct_over, 1)

# Equipo con mejor diferencia de goles por mundial
for wc in [2018, 2022, 2026]:
    subset = partidos[partidos["mundial"] == wc]

    # Calcular GF y GC por equipo (como local y visitante)
    local = subset.groupby("equipo_local").agg(
        gf=("goles_local", "sum"), gc=("goles_visitante", "sum")
    )
    visita = subset.groupby("equipo_visitante").agg(
        gf=("goles_visitante", "sum"), gc=("goles_local", "sum")
    )
    total_eq = local.add(visita, fill_value=0)
    total_eq["dg"] = total_eq["gf"] - total_eq["gc"]
    mejor = total_eq["dg"].idxmax()
    comparacion.loc[comparacion["mundial"] == wc, "mejor_dif_goles"] = f"{mejor} (+{int(total_eq.loc[mejor, 'dg'])})"

print("\n  Tabla comparativa de los tres mundiales:\n")
print(comparacion.to_string(index=False))

# ── Gráfico 1: Goles totales vs Goles por partido ──

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Goles totales
axes[0].bar(
    comparacion["mundial"].astype(str),
    comparacion["goles_totales"],
    color=["#3498db", "#e74c3c", "#2ecc71"],
    edgecolor="white",
    linewidth=1.5,
)
axes[0].set_title("Goles Totales por Mundial", fontsize=14, fontweight="bold")
axes[0].set_ylabel("Goles")
for i, v in enumerate(comparacion["goles_totales"]):
    axes[0].text(i, v + 2, str(int(v)), ha="center", fontweight="bold", fontsize=12)

# Goles por partido
axes[1].bar(
    comparacion["mundial"].astype(str),
    comparacion["goles_por_partido"],
    color=["#3498db", "#e74c3c", "#2ecc71"],
    edgecolor="white",
    linewidth=1.5,
)
axes[1].set_title("Goles por Partido (tasa normalizada)", fontsize=14, fontweight="bold")
axes[1].set_ylabel("Goles / Partido")
for i, v in enumerate(comparacion["goles_por_partido"]):
    axes[1].text(i, v + 0.05, f"{v:.2f}", ha="center", fontweight="bold", fontsize=12)

plt.tight_layout()
plt.savefig(DATA / "grafico_goles.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✓ Gráfico guardado: datos/grafico_goles.png")

# ── Gráfico 2: Distribución de resultados ──

resultados_por_mundial = partidos.groupby(["mundial", "resultado_local"]).size().unstack(fill_value=0)
# Reordenar columnas
for col in ["Gana", "Empata", "Pierde"]:
    if col not in resultados_por_mundial.columns:
        resultados_por_mundial[col] = 0
resultados_por_mundial = resultados_por_mundial[["Gana", "Empata", "Pierde"]]

# Convertir a porcentaje
resultados_pct = resultados_por_mundial.div(resultados_por_mundial.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(10, 5))
resultados_pct.plot(
    kind="bar",
    ax=ax,
    color=["#2ecc71", "#f39c12", "#e74c3c"],
    edgecolor="white",
    linewidth=1.5,
)
ax.set_title("Distribución de Resultados (% por Mundial)", fontsize=14, fontweight="bold")
ax.set_ylabel("Porcentaje (%)")
ax.set_xlabel("Mundial")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(["Victoria Local", "Empate", "Victoria Visitante"], loc="upper right")

# Etiquetas
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f%%", fontsize=9, padding=2)

plt.tight_layout()
plt.savefig(DATA / "grafico_resultados.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✓ Gráfico guardado: datos/grafico_resultados.png")

# ── Tabla por equipo ──

print("\n" + "─" * 60)
print("  TABLA POR EQUIPO")
print("─" * 60)


def tabla_por_equipo(df):
    """Crea tabla de posiciones con PJ, PG, PE, PP, GF, GC, DG, PTS."""
    filas = []

    for _, r in df.iterrows():
        gl = int(r["goles_local"])
        gv = int(r["goles_visitante"])
        wc = r["mundial"]

        # Fila para el equipo local
        filas.append({
            "mundial": wc,
            "equipo": r["equipo_local"],
            "grupo": r["grupo"],
            "pj": 1,
            "pg": 1 if gl > gv else 0,
            "pe": 1 if gl == gv else 0,
            "pp": 1 if gl < gv else 0,
            "gf": gl,
            "gc": gv,
        })
        # Fila para el equipo visitante
        filas.append({
            "mundial": wc,
            "equipo": r["equipo_visitante"],
            "grupo": r["grupo"],
            "pj": 1,
            "pg": 1 if gv > gl else 0,
            "pe": 1 if gv == gl else 0,
            "pp": 1 if gv < gl else 0,
            "gf": gv,
            "gc": gl,
        })

    tabla = pd.DataFrame(filas)
    tabla = tabla.groupby(["mundial", "grupo", "equipo"]).sum().reset_index()
    tabla["dg"] = tabla["gf"] - tabla["gc"]
    tabla["pts"] = tabla["pg"] * 3 + tabla["pe"]
    tabla["pts_por_partido"] = (tabla["pts"] / tabla["pj"]).round(2)
    tabla = tabla.sort_values(["mundial", "grupo", "pts", "dg", "gf"],
                              ascending=[True, True, False, False, False])
    return tabla


tabla_equipos = tabla_por_equipo(partidos)
print(f"\n  {len(tabla_equipos)} registros de equipo (equipo × mundial)")

# Mostrar tabla de cada mundial con los mejores
for wc in [2018, 2022, 2026]:
    sub = tabla_equipos[tabla_equipos["mundial"] == wc]
    top5 = sub.nlargest(5, "pts")
    print(f"\n  Top 5 equipos — Mundial {wc}:")
    print(top5[["equipo", "grupo", "pj", "pg", "pe", "pp", "gf", "gc", "dg", "pts"]].to_string(index=False))

# Pregunta central
print("\n  ── Pregunta central ──")
print("  ¿Cambian las conclusiones cuando se comparan totales vs tasas?")
print(f"  → Goles totales: 2018={int(comparacion.loc[comparacion['mundial']==2018, 'goles_totales'].values[0])}, "
      f"2022={int(comparacion.loc[comparacion['mundial']==2022, 'goles_totales'].values[0])}, "
      f"2026={int(comparacion.loc[comparacion['mundial']==2026, 'goles_totales'].values[0])}")
print(f"  → Goles/partido: 2018={comparacion.loc[comparacion['mundial']==2018, 'goles_por_partido'].values[0]}, "
      f"2022={comparacion.loc[comparacion['mundial']==2022, 'goles_por_partido'].values[0]}, "
      f"2026={comparacion.loc[comparacion['mundial']==2026, 'goles_por_partido'].values[0]}")
print("  → SÍ cambian. Al comparar totales, 2026 parece tener más goles, pero")
print("    tiene 72 partidos vs 48. Las tasas normalizadas dan la imagen real.")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  PARTE D — CLASIFICACIÓN SUPERVISADA                               ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\n\n" + "=" * 70)
print("  PARTE D — CLASIFICACIÓN SUPERVISADA")
print("=" * 70)

# ── Variables previas al partido ──

print("\n  Construyendo variables previas al partido...")


def construir_variables_previas(partidos_df):
    """Construye features pre-partido para cada registro.
    Para cada equipo se mantiene un acumulador de PJ, pts, GF, GC
    y se guardan los promedios ANTES de actualizar con el partido actual.
    """
    registros = []

    for wc in sorted(partidos_df["mundial"].unique()):
        sub = partidos_df[partidos_df["mundial"] == wc].sort_values(
            ["jornada", "fecha", "partido_id"]
        ).copy()

        # Acumuladores por equipo dentro de este torneo
        acum = {}  # equipo → {"pj": int, "pts": int, "gf": int, "gc": int}

        for _, r in sub.iterrows():
            local = r["equipo_local"]
            visitante = r["equipo_visitante"]
            gl = int(r["goles_local"])
            gv = int(r["goles_visitante"])

            # Obtener estadísticas PREVIAS (antes del partido actual)
            stats_l = acum.get(local, {"pj": 0, "pts": 0, "gf": 0, "gc": 0})
            stats_v = acum.get(visitante, {"pj": 0, "pts": 0, "gf": 0, "gc": 0})

            # Promedios previos del local
            if stats_l["pj"] > 0:
                local_pts_prom = stats_l["pts"] / stats_l["pj"]
                local_gf_prom = stats_l["gf"] / stats_l["pj"]
                local_gc_prom = stats_l["gc"] / stats_l["pj"]
                local_gd_prom = (stats_l["gf"] - stats_l["gc"]) / stats_l["pj"]
            else:
                local_pts_prom = 0.0
                local_gf_prom = 0.0
                local_gc_prom = 0.0
                local_gd_prom = 0.0

            # Promedios previos del visitante
            if stats_v["pj"] > 0:
                visita_pts_prom = stats_v["pts"] / stats_v["pj"]
                visita_gf_prom = stats_v["gf"] / stats_v["pj"]
                visita_gc_prom = stats_v["gc"] / stats_v["pj"]
                visita_gd_prom = (stats_v["gf"] - stats_v["gc"]) / stats_v["pj"]
            else:
                visita_pts_prom = 0.0
                visita_gf_prom = 0.0
                visita_gc_prom = 0.0
                visita_gd_prom = 0.0

            registros.append({
                "partido_id": r["partido_id"],
                "mundial": wc,
                "jornada": int(r["jornada"]),
                "equipo_local": local,
                "equipo_visitante": visitante,
                "goles_local": gl,
                "goles_visitante": gv,
                "diferencia_goles": gl - gv,
                "resultado_local": r["resultado_local"],
                "local_es_anfitrion": int(r["local_es_anfitrion"]),
                "visitante_es_anfitrion": int(r["visitante_es_anfitrion"]),
                "local_pts_prom_pre": round(local_pts_prom, 3),
                "local_gf_prom_pre": round(local_gf_prom, 3),
                "local_gc_prom_pre": round(local_gc_prom, 3),
                "local_gd_prom_pre": round(local_gd_prom, 3),
                "visita_pts_prom_pre": round(visita_pts_prom, 3),
                "visita_gf_prom_pre": round(visita_gf_prom, 3),
                "visita_gc_prom_pre": round(visita_gc_prom, 3),
                "visita_gd_prom_pre": round(visita_gd_prom, 3),
            })

            # Actualizar acumuladores DESPUÉS de guardar
            # Puntos del local
            pts_l = 3 if gl > gv else (1 if gl == gv else 0)
            pts_v = 3 if gv > gl else (1 if gv == gl else 0)

            if local not in acum:
                acum[local] = {"pj": 0, "pts": 0, "gf": 0, "gc": 0}
            acum[local]["pj"] += 1
            acum[local]["pts"] += pts_l
            acum[local]["gf"] += gl
            acum[local]["gc"] += gv

            if visitante not in acum:
                acum[visitante] = {"pj": 0, "pts": 0, "gf": 0, "gc": 0}
            acum[visitante]["pj"] += 1
            acum[visitante]["pts"] += pts_v
            acum[visitante]["gf"] += gv
            acum[visitante]["gc"] += gl

    return pd.DataFrame(registros)


features_df = construir_variables_previas(partidos)
print(f"  ✓ Features construidos: {features_df.shape[0]} filas × {features_df.shape[1]} columnas")

# ── Entrenar modelo ──

features_cols = [
    "jornada",
    "local_pts_prom_pre", "visita_pts_prom_pre",
    "local_gd_prom_pre", "visita_gd_prom_pre",
    "local_gf_prom_pre", "visita_gf_prom_pre",
    "local_es_anfitrion", "visitante_es_anfitrion",
]

target = "resultado_local"

# Separar entrenamiento (2018 + 2022) y prueba (2026)
train = features_df[features_df["mundial"].isin([2018, 2022])].copy()
test = features_df[features_df["mundial"] == 2026].copy()

X_train = train[features_cols].fillna(0)
y_train = train[target]
X_test = test[features_cols].fillna(0)
y_test = test[target]

print(f"\n  Entrenamiento: {len(train)} partidos (2018 + 2022)")
print(f"  Prueba:        {len(test)} partidos (2026)")

# Línea base: clase más frecuente
clase_mas_frecuente = y_train.value_counts().idxmax()
baseline_acc = (y_test == clase_mas_frecuente).mean()
print(f"\n  Línea base (siempre predecir '{clase_mas_frecuente}'): {baseline_acc:.1%}")

# Entrenar DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"  Accuracy del árbol de decisión: {accuracy:.1%}")
print(f"  {'✓ Supera' if accuracy > baseline_acc else '✗ No supera'} la línea base")

# Distribución de predicciones
print(f"\n  Distribución real en test:")
for clase, n in y_test.value_counts().items():
    print(f"    {clase}: {n} ({n/len(y_test):.1%})")

print(f"\n  Distribución predicha:")
pred_series = pd.Series(y_pred)
for clase, n in pred_series.value_counts().items():
    print(f"    {clase}: {n} ({n/len(y_pred):.1%})")

# Importancia de features
print(f"\n  Importancia de variables:")
importancias = pd.Series(clf.feature_importances_, index=features_cols).sort_values(ascending=False)
for feat, imp in importancias.items():
    if imp > 0:
        print(f"    {feat}: {imp:.3f}")

# Matriz de confusión
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, ax=axes[0],
    cmap="Blues",
    display_labels=["Empata", "Gana", "Pierde"],
)
axes[0].set_title("Matriz de Confusión — Sin Fuga", fontsize=13, fontweight="bold")

# ── Experimento de fuga de información ──

print("\n" + "─" * 60)
print("  EXPERIMENTO: FUGA DE INFORMACIÓN (DATA LEAKAGE)")
print("─" * 60)

features_fuga = features_cols + ["goles_local", "goles_visitante", "diferencia_goles"]

X_train_fuga = train[features_fuga].fillna(0)
X_test_fuga = test[features_fuga].fillna(0)

clf_fuga = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42)
clf_fuga.fit(X_train_fuga, y_train)

y_pred_fuga = clf_fuga.predict(X_test_fuga)
accuracy_fuga = accuracy_score(y_test, y_pred_fuga)

print(f"\n  Accuracy SIN fuga:  {accuracy:.1%}")
print(f"  Accuracy CON fuga:  {accuracy_fuga:.1%}")
print(f"\n  ⚠ La precisión subió a {accuracy_fuga:.1%} porque el modelo tiene acceso a")
print(f"    los goles finales del partido (goles_local, goles_visitante, diferencia_goles),")
print(f"    que son información del RESULTADO mismo. Esto es fuga de información:")
print(f"    el modelo no está prediciendo, está LEYENDO la respuesta de la variable objetivo.")
print(f"    En un escenario real, no conoceríamos estos valores antes del partido.")

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_fuga, ax=axes[1],
    cmap="Reds",
    display_labels=["Empata", "Gana", "Pierde"],
)
axes[1].set_title("Matriz de Confusión — CON Fuga ⚠", fontsize=13, fontweight="bold")

plt.tight_layout()
plt.savefig(DATA / "grafico_confusion.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✓ Gráfico guardado: datos/grafico_confusion.png")

# Importancia con fuga
print(f"\n  Importancia de variables CON fuga:")
importancias_fuga = pd.Series(clf_fuga.feature_importances_, index=features_fuga).sort_values(ascending=False)
for feat, imp in importancias_fuga.items():
    if imp > 0:
        print(f"    {feat}: {imp:.3f}")

# ── Visualizar el árbol ──

fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(
    clf,
    feature_names=features_cols,
    class_names=["Empata", "Gana", "Pierde"],
    filled=True,
    rounded=True,
    fontsize=8,
    ax=ax,
)
ax.set_title("Árbol de Decisión (sin fuga)", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(DATA / "grafico_arbol.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✓ Gráfico guardado: datos/grafico_arbol.png")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  REFLEXIÓN FINAL                                                    ║
# ╚══════════════════════════════════════════════════════════════════════╝

print("\n\n" + "=" * 70)
print("  REFLEXIÓN FINAL")
print("=" * 70)

print("""
  1. ¿Qué problema de calidad fue el más difícil?
     → Los formatos de fecha (6 formatos distintos incluyendo seriales de Excel
       y fechas ambiguas DD/MM vs MM/DD). También la reparación de goles cuando
       los valores de columna y el marcador se contradicen.

  2. ¿Qué decisión de limpieza podría cambiar los resultados?
     → La priorización del marcador sobre los goles individuales. Si se
       prioriza al revés, algunos partidos tendrían resultados diferentes.
       También el manejo de goles negativos (se tratan como inválidos).

  3. ¿Por qué 2026 debe compararse mediante tasas?
     → Porque 2026 tiene 48 equipos y 72 partidos de fase de grupos,
       mientras que 2018 y 2022 tienen 32 equipos y 48 partidos.
       Los totales crudos no son comparables; las tasas (goles/partido,
       porcentajes) normalizan la diferencia de escala.

  4. ¿El árbol supera la línea base?
     → Con solo 3 jornadas por equipo y variables pre-partido limitadas,
       el modelo tiene poca información. Es probable que supere marginalmente
       la línea base o quede cerca de ella. Esto es esperable.

  5. ¿Qué variables reales agregarías para mejorar una predicción?
     → Ranking FIFA, historial de enfrentamientos, valor de mercado del plantel,
       lesiones de jugadores clave, clima, distancia viajada, y días de descanso.

  6. ¿Por qué un resultado de 100% puede ser una señal de alarma?
     → Porque indica fuga de información (data leakage). Si el modelo tiene
       acceso a datos que contienen la respuesta (como goles finales),
       no está prediciendo sino leyendo. En producción esos datos no
       estarían disponibles antes del partido.
""")

# ── Bitácora de decisiones de limpieza ──

print("=" * 70)
print("  BITÁCORA DE DECISIONES DE LIMPIEZA")
print("=" * 70)

print("""
  ┌─────────────────────────────────────────────────────────────────┐
  │  INCIDENCIA                           │  DECISIÓN              │
  ├─────────────────────────────────────────────────────────────────┤
  │  3 esquemas de columnas distintos     │  Mapas de renombrado   │
  │  6+ formatos de fecha                 │  Parseo con fallbacks  │
  │  Serial Excel (46187)                 │  EXCEL_EPOCH + delta   │
  │  Fecha "N/D"                          │  pd.NaT                │
  │  5 separadores de marcador            │  Regex → formato X-Y   │
  │  "sin dato" como marcador             │  NaN, inferir de goles │
  │  "5 goles", "1 goles"                 │  Regex: extraer dígito │
  │  Goles negativos (-1, -2)             │  Inválido → del marcad.│
  │  Goles N/A, s/d, vacíos              │  Inferir del marcador  │
  │  Espacios en nombres (" russia ")     │  strip()               │
  │  40+ variantes de equipos             │  Catálogo + clave norm.│
  │  Fase: 8 variantes                    │  Todo = "Fase de grupo"│
  │  Grupo: "group-a", "A ", "s/d"       │  Regex → letra A-L     │
  │  Grupo vacío (M-2018-10, etc.)       │  Inferir por equipos   │
  │  Anfitrión: Sí/si/TRUE/1/0/FALSE    │  normalizar_booleano() │
  │  2026: 2 cols anfitrión              │  host_h + host_a       │
  │  8 filas duplicadas (3 archivos)     │  drop_duplicates first │
  │  Conflicto goles vs marcador         │  Marcador tiene priorid│
  └─────────────────────────────────────────────────────────────────┘
""")

print("  ✓ Script completado exitosamente.")
print("  ✓ Archivos generados:")
print("    → datos/base_limpia_168.csv")
print("    → datos/grafico_goles.png")
print("    → datos/grafico_resultados.png")
print("    → datos/grafico_confusion.png")
print("    → datos/grafico_arbol.png")
