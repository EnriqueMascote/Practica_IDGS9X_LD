"""
Limpieza y análisis de la fase de grupos de los Mundiales 2018, 2022 y 2026.

Reproduce en Python lo pedido en Guia.md (Parte C y D):
- perfilado de las tres fuentes sucias;
- unificación de esquemas y de nombres de equipo vía catálogo;
- limpieza de fechas (incluye seriales de Excel), grupos, booleanos y marcador;
- validaciones obligatorias sobre la base integrada;
- comparación entre torneos y tabla por equipo;
- variables previas al partido (sin fuga de información) y un árbol de
  decisión sencillo, con el experimento de fuga para contraste.
"""
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'datos'
SALIDA = BASE / 'salida'
SALIDA.mkdir(exist_ok=True)

pd.set_option('display.max_columns', 50)

RANGOS_FECHA = {
    2018: ('2018-06-14', '2018-06-28'),
    2022: ('2022-11-20', '2022-12-02'),
    2026: ('2026-06-11', '2026-06-27'),
}
PARTIDOS_ESPERADOS = {2018: 48, 2022: 48, 2026: 72}

RENAME_MAPS = {
    2018: {
        'ID Partido': 'partido_id', 'Fase': 'fase', 'Grupo': 'grupo',
        'Jornada': 'jornada', 'Fecha': 'fecha', 'Equipo Local': 'equipo_local',
        'Equipo Visitante': 'equipo_visitante', 'Goles Local': 'goles_local',
        'Goles Visitante': 'goles_visitante', 'Marcador': 'marcador',
        'Anfitrión Local': 'local_es_anfitrion', 'Fuente': 'fuente',
    },
    2022: {
        'match_id': 'partido_id', 'stage': 'fase', 'group_name': 'grupo',
        'match_day': 'jornada', 'date': 'fecha', 'local': 'equipo_local',
        'visitor': 'equipo_visitante', 'home_score': 'goles_local',
        'away_score': 'goles_visitante', 'score_text': 'marcador',
        'home_host': 'local_es_anfitrion', 'source_url': 'fuente',
    },
    2026: {
        'match': 'partido_id', 'round': 'fase', 'grp': 'grupo', 'md': 'jornada',
        'played_on': 'fecha', 'home': 'equipo_local', 'away': 'equipo_visitante',
        'HG': 'goles_local', 'AG': 'goles_visitante', 'result_raw': 'marcador',
        'host_h': 'local_es_anfitrion', 'host_a': 'visitante_es_anfitrion',
        'source': 'fuente',
    },
}

COLUMNAS_BASE = [
    'partido_id', 'mundial', 'fase', 'grupo', 'jornada', 'fecha',
    'equipo_local', 'equipo_visitante', 'goles_local', 'goles_visitante',
    'marcador', 'local_es_anfitrion', 'visitante_es_anfitrion', 'fuente',
]

FORMATOS_FECHA = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%y', '%b %d, %Y']
SEPARADOR_MARCADOR = re.compile(r'\s*[-–—:xX]\s*')
VALORES_VERDADEROS = {'si', 'sí', 'true', '1', 'yes'}

incidencias = []  # bitácora de problemas detectados y decisiones de limpieza


def log(mensaje):
    incidencias.append(mensaje)


# ---------------------------------------------------------------------------
# Perfilado (Parte C, sección "Perfilado")
# ---------------------------------------------------------------------------

def perfil(df, nombre):
    print(f'\n=== Perfil {nombre} ===')
    print('Dimensiones:', df.shape)
    print('Nulos por columna:')
    print(df.isna().sum())
    print('Filas totalmente duplicadas:', df.duplicated().sum())

    for col in df.columns:
        etiqueta = col.lower()
        if any(pista in etiqueta for pista in ('grupo', 'group', 'grp', 'fase', 'stage', 'round')):
            print(f'Valores únicos en {col!r}:', sorted(df[col].dropna().unique().tolist()))

    for col in df.columns:
        if any(pista in col.lower() for pista in ('gol', 'score')) or col in ('HG', 'AG'):
            numerico = pd.to_numeric(df[col], errors='coerce')
            no_convertible = df[col][numerico.isna() & df[col].notna()]
            if len(no_convertible):
                print(f'{col!r}: valores que no son número directo ->', no_convertible.unique().tolist())


# ---------------------------------------------------------------------------
# Catálogo de equipos (Parte C, "Normalizar equipos")
# ---------------------------------------------------------------------------

def clave_texto(valor):
    if pd.isna(valor):
        return ''
    texto = unicodedata.normalize('NFKD', str(valor).strip().lower())
    texto = ''.join(c for c in texto if not unicodedata.combining(c))
    return re.sub(r'[^a-z0-9]+', ' ', texto).strip()


catalogo = pd.read_csv(DATA / 'catalogo_equipos.csv')
MAPA_EQUIPOS = {
    clave_texto(variante): canonico
    for variante, canonico in zip(catalogo['variante'], catalogo['nombre_canonico'])
}


def normalizar_equipo(valor):
    clave = clave_texto(valor)
    if clave in MAPA_EQUIPOS:
        return MAPA_EQUIPOS[clave]
    log(f"Equipo sin coincidencia en el catálogo: '{valor}' (clave='{clave}').")
    return str(valor).strip() if pd.notna(valor) else np.nan


# ---------------------------------------------------------------------------
# Fechas, grupos, booleanos y marcador (Parte C, "Fechas, grupos...")
# ---------------------------------------------------------------------------

def convertir_fecha(valor, mundial):
    if pd.isna(valor):
        return pd.NaT
    texto = str(valor).strip()
    if texto == '' or texto.lower() in ('nan', 'n/d', 's/d'):
        return pd.NaT

    lo, hi = (pd.Timestamp(x) for x in RANGOS_FECHA[mundial])

    # 1. Serial de Excel: un entero de días desde el origen 1899-12-30.
    if re.fullmatch(r'\d+', texto):
        fecha = pd.Timestamp('1899-12-30') + pd.to_timedelta(int(texto), unit='D')
        if not (lo <= fecha <= hi):
            log(f'Fecha serial de Excel fuera de rango para {mundial}: {texto} -> {fecha.date()}.')
        return fecha

    # 2. Varios formatos de texto; se prefiere el que cae dentro del rango del torneo.
    candidatas = []
    for fmt in FORMATOS_FECHA:
        fecha = pd.to_datetime(texto, format=fmt, errors='coerce')
        if pd.notna(fecha):
            candidatas.append(fecha)

    for fecha in candidatas:
        if lo <= fecha <= hi:
            return fecha

    if candidatas:
        log(f"Fecha '{texto}' ({mundial}): ningún formato cae en el rango del torneo; se usa {candidatas[0].date()}.")
        return candidatas[0]

    log(f"Fecha no reconocida para {mundial}: '{texto}'.")
    return pd.NaT


def extraer_numero(valor):
    if pd.isna(valor):
        return np.nan
    coincidencia = re.search(r'-?\d+', str(valor))
    return float(coincidencia.group()) if coincidencia else np.nan


def separar_marcador(valor):
    if pd.isna(valor):
        return np.nan, np.nan
    partes = [p for p in SEPARADOR_MARCADOR.split(str(valor).strip()) if p != '']
    if len(partes) != 2:
        return np.nan, np.nan
    try:
        return float(partes[0]), float(partes[1])
    except ValueError:
        return np.nan, np.nan


def normalizar_grupo(valor):
    if pd.isna(valor):
        return np.nan
    coincidencia = re.search(r'\b([A-L])\b', str(valor).strip().upper())
    return coincidencia.group(1) if coincidencia else np.nan


def normalizar_booleano(valor):
    if pd.isna(valor):
        return False
    return str(valor).strip().lower() in VALORES_VERDADEROS


# ---------------------------------------------------------------------------
# Función de limpieza reproducible (Parte C)
# ---------------------------------------------------------------------------

def limpiar_mundial(df_crudo, mundial):
    df = df_crudo.rename(columns=RENAME_MAPS[mundial])
    for columna in COLUMNAS_BASE:
        if columna not in df.columns:
            df[columna] = np.nan
    df = df[COLUMNAS_BASE].copy()

    df['partido_id'] = df['partido_id'].str.strip()
    df['mundial'] = mundial
    # El kit solo contiene fase de grupos; las variantes de "fase" no aportan información nueva.
    df['fase'] = 'Fase de grupos'
    df['fuente'] = df['fuente'].astype(str).str.strip()

    duplicados = df.duplicated(subset=['partido_id'], keep='first')
    if duplicados.any():
        log(f'{mundial}: se eliminaron {int(duplicados.sum())} filas duplicadas '
            f"({df.loc[duplicados, 'partido_id'].tolist()}).")
    df = df.loc[~duplicados].copy()

    df['equipo_local'] = df['equipo_local'].apply(normalizar_equipo)
    df['equipo_visitante'] = df['equipo_visitante'].apply(normalizar_equipo)

    df['grupo'] = df['grupo'].apply(normalizar_grupo)
    df['jornada'] = df['jornada'].apply(extraer_numero)
    df['fecha'] = df['fecha'].apply(lambda v: convertir_fecha(v, mundial))

    df['local_es_anfitrion'] = df['local_es_anfitrion'].apply(normalizar_booleano)
    df['visitante_es_anfitrion'] = df['visitante_es_anfitrion'].apply(normalizar_booleano)
    if mundial in (2018, 2022):
        log(f'{mundial}: la fuente no distingue anfitrión del visitante; se asume False '
            '(en estos datos el anfitrión siempre figura como equipo local).')

    # Reparar goles con el marcador (Paso 7 de la guía).
    goles_local_bruto = df['goles_local'].apply(extraer_numero)
    goles_visita_bruto = df['goles_visitante'].apply(extraer_numero)
    marcador_local, marcador_visita = zip(*df['marcador'].apply(separar_marcador))
    marcador_local = pd.Series(marcador_local, index=df.index)
    marcador_visita = pd.Series(marcador_visita, index=df.index)
    tiene_marcador = marcador_local.notna() & marcador_visita.notna()

    conflicto = (
        tiene_marcador & goles_local_bruto.notna() & goles_visita_bruto.notna()
        & ((marcador_local != goles_local_bruto) | (marcador_visita != goles_visita_bruto))
    )
    for pid in df.loc[conflicto, 'partido_id']:
        log(f'Partido {pid}: el marcador y los goles por separado no coinciden; se usó el marcador.')

    goles_local = marcador_local.where(tiene_marcador, goles_local_bruto)
    goles_visitante = marcador_visita.where(tiene_marcador, goles_visita_bruto)

    negativos = ((goles_local < 0) | (goles_visitante < 0)).fillna(False)
    for pid in df.loc[negativos, 'partido_id']:
        log(f'Partido {pid}: goles negativos detectados; se descartan como inválidos.')
    df['goles_local'] = goles_local.mask(negativos)
    df['goles_visitante'] = goles_visitante.mask(negativos)

    validos = df['goles_local'].notna() & df['goles_visitante'].notna()
    df['marcador'] = np.where(
        validos,
        df['goles_local'].fillna(0).astype(int).astype(str) + '-' + df['goles_visitante'].fillna(0).astype(int).astype(str),
        np.nan,
    )

    # Inferir grupos faltantes a partir de los equipos que sí tienen grupo conocido.
    equipo_a_grupo = {}
    for _, fila in df.dropna(subset=['grupo']).iterrows():
        equipo_a_grupo.setdefault(fila['equipo_local'], fila['grupo'])
        equipo_a_grupo.setdefault(fila['equipo_visitante'], fila['grupo'])

    def inferir_grupo(fila):
        if pd.notna(fila['grupo']):
            return fila['grupo']
        grupo = equipo_a_grupo.get(fila['equipo_local']) or equipo_a_grupo.get(fila['equipo_visitante'])
        if grupo:
            log(f"Partido {fila['partido_id']}: grupo inferido ({grupo}) a partir de los equipos.")
        else:
            log(f"Partido {fila['partido_id']}: no fue posible inferir el grupo.")
        return grupo

    df['grupo'] = df.apply(inferir_grupo, axis=1)

    df['goles_totales'] = df['goles_local'] + df['goles_visitante']
    df['diferencia_goles'] = df['goles_local'] - df['goles_visitante']

    resultado = np.select(
        [df['goles_local'] > df['goles_visitante'], df['goles_local'] == df['goles_visitante']],
        ['Gana', 'Empata'],
        default='Pierde',
    )
    resultado = pd.Series(resultado, index=df.index)
    resultado[~validos] = np.nan
    df['resultado_local'] = resultado

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Validaciones obligatorias
# ---------------------------------------------------------------------------

def validar(partidos):
    print('\n=== Validaciones ===')
    conteo = partidos.groupby('mundial').size().to_dict()
    for mundial, esperado in PARTIDOS_ESPERADOS.items():
        obtenido = conteo.get(mundial, 0)
        print(f'{mundial}: {obtenido} partidos (esperado {esperado}) -> '
              f"{'OK' if obtenido == esperado else 'FALLA'}")

    print('Duplicados por partido_id:', int(partidos.duplicated(subset=['partido_id']).sum()))
    print('Goles negativos:', int(((partidos['goles_local'] < 0) | (partidos['goles_visitante'] < 0)).sum()))

    nulos = partidos[['equipo_local', 'equipo_visitante', 'goles_local', 'goles_visitante', 'grupo']].isna().sum()
    print('Nulos en columnas clave:')
    print(nulos)

    marcador_calc = np.where(
        partidos['goles_local'].notna() & partidos['goles_visitante'].notna(),
        partidos['goles_local'].fillna(0).astype(int).astype(str) + '-'
        + partidos['goles_visitante'].fillna(0).astype(int).astype(str),
        np.nan,
    )
    print('Marcador inconsistente con goles:', int((marcador_calc != partidos['marcador']).sum()))
    print('Partidos con más de una fila:', int((partidos.groupby('partido_id').size() > 1).sum()))


# ---------------------------------------------------------------------------
# Comparación de los Mundiales y tabla por equipo (Parte C / Parte B)
# ---------------------------------------------------------------------------

def comparar_torneos(partidos):
    resumen = partidos.groupby('mundial').agg(partidos=('partido_id', 'count'), goles=('goles_totales', 'sum'))
    resumen['goles_por_partido'] = resumen['goles'] / resumen['partidos']
    resumen['empates'] = partidos.groupby('mundial')['resultado_local'].apply(lambda s: (s == 'Empata').sum())
    resumen['porcentaje_empates'] = resumen['empates'] / resumen['partidos'] * 100

    gano_anfitrion = (
        (partidos['local_es_anfitrion'] & (partidos['resultado_local'] == 'Gana'))
        | (partidos['visitante_es_anfitrion'] & (partidos['resultado_local'] == 'Pierde'))
    )
    resumen['porcentaje_victorias_anfitrion'] = gano_anfitrion.groupby(partidos['mundial']).mean() * 100
    resumen['porcentaje_mas_2_5_goles'] = (partidos['goles_totales'] > 2.5).groupby(partidos['mundial']).mean() * 100

    print('\n=== Comparación de torneos ===')
    print(resumen)

    fig, ax = plt.subplots()
    resumen[['goles', 'goles_por_partido']].plot(kind='bar', ax=ax, secondary_y='goles_por_partido',
                                                  title='Goles totales vs. goles por partido')
    fig.tight_layout()
    fig.savefig(SALIDA / 'goles_por_partido.png')
    plt.close(fig)

    fig, ax = plt.subplots()
    partidos.groupby(['mundial', 'resultado_local']).size().unstack().plot(
        kind='bar', stacked=True, ax=ax, title='Distribución de resultados por Mundial')
    fig.tight_layout()
    fig.savefig(SALIDA / 'distribucion_resultados.png')
    plt.close(fig)

    return resumen


def tabla_equipos(partidos):
    local = partidos.rename(columns={
        'equipo_local': 'equipo', 'goles_local': 'gf', 'goles_visitante': 'gc',
    })[['mundial', 'equipo', 'gf', 'gc', 'resultado_local']].copy()
    local['resultado'] = local['resultado_local']

    visita = partidos.rename(columns={
        'equipo_visitante': 'equipo', 'goles_visitante': 'gf', 'goles_local': 'gc',
    })[['mundial', 'equipo', 'gf', 'gc', 'resultado_local']].copy()
    visita['resultado'] = visita['resultado_local'].map({'Gana': 'Pierde', 'Pierde': 'Gana', 'Empata': 'Empata'})

    apariciones = pd.concat([local, visita], ignore_index=True).drop(columns='resultado_local')
    apariciones['pts'] = apariciones['resultado'].map({'Gana': 3, 'Empata': 1, 'Pierde': 0})

    tabla = apariciones.groupby(['mundial', 'equipo']).agg(
        pj=('resultado', 'count'),
        pg=('resultado', lambda s: (s == 'Gana').sum()),
        pe=('resultado', lambda s: (s == 'Empata').sum()),
        pp=('resultado', lambda s: (s == 'Pierde').sum()),
        gf=('gf', 'sum'),
        gc=('gc', 'sum'),
        pts=('pts', 'sum'),
    ).reset_index()
    tabla['dg'] = tabla['gf'] - tabla['gc']
    tabla['pts_por_partido'] = tabla['pts'] / tabla['pj']
    return tabla.sort_values(['mundial', 'pts', 'dg'], ascending=[True, False, False])


# ---------------------------------------------------------------------------
# Variables previas al partido (Parte D) — sin fuga de información
# ---------------------------------------------------------------------------

def construir_variables_previas(partidos):
    partidos = partidos.sort_values(['mundial', 'fecha', 'jornada', 'partido_id']).reset_index(drop=True)
    estado = {}
    filas = []

    def promedios(e):
        if e['pj'] == 0:
            return 0.0, 0.0, 0.0
        return e['pts'] / e['pj'], (e['gf'] - e['gc']) / e['pj'], e['gf'] / e['pj']

    for _, partido in partidos.iterrows():
        clave_local = (partido['mundial'], partido['equipo_local'])
        clave_visita = (partido['mundial'], partido['equipo_visitante'])
        est_local = estado.setdefault(clave_local, {'pj': 0, 'pts': 0, 'gf': 0, 'gc': 0})
        est_visita = estado.setdefault(clave_visita, {'pj': 0, 'pts': 0, 'gf': 0, 'gc': 0})

        local_pts_prom_pre, local_gd_prom_pre, local_gf_prom_pre = promedios(est_local)
        visita_pts_prom_pre, visita_gd_prom_pre, visita_gf_prom_pre = promedios(est_visita)

        filas.append({
            'partido_id': partido['partido_id'],
            'mundial': partido['mundial'],
            'jornada': partido['jornada'],
            'local_pts_prom_pre': local_pts_prom_pre,
            'visita_pts_prom_pre': visita_pts_prom_pre,
            'local_gd_prom_pre': local_gd_prom_pre,
            'visita_gd_prom_pre': visita_gd_prom_pre,
            'local_gf_prom_pre': local_gf_prom_pre,
            'visita_gf_prom_pre': visita_gf_prom_pre,
            'local_es_anfitrion': int(partido['local_es_anfitrion']),
            'visitante_es_anfitrion': int(partido['visitante_es_anfitrion']),
            'goles_local': partido['goles_local'],
            'goles_visitante': partido['goles_visitante'],
            'diferencia_goles': partido['diferencia_goles'],
            'resultado_local': partido['resultado_local'],
        })

        pts_local = 3 if partido['resultado_local'] == 'Gana' else (1 if partido['resultado_local'] == 'Empata' else 0)
        pts_visita = 3 if partido['resultado_local'] == 'Pierde' else (1 if partido['resultado_local'] == 'Empata' else 0)

        est_local['pj'] += 1
        est_local['pts'] += pts_local
        est_local['gf'] += partido['goles_local']
        est_local['gc'] += partido['goles_visitante']

        est_visita['pj'] += 1
        est_visita['pts'] += pts_visita
        est_visita['gf'] += partido['goles_visitante']
        est_visita['gc'] += partido['goles_local']

    return pd.DataFrame(filas).dropna(subset=['resultado_local'])


FEATURES = [
    'jornada',
    'local_pts_prom_pre', 'visita_pts_prom_pre',
    'local_gd_prom_pre', 'visita_gd_prom_pre',
    'local_gf_prom_pre', 'visita_gf_prom_pre',
    'local_es_anfitrion', 'visitante_es_anfitrion',
]


def entrenar_y_evaluar(features_df, columnas_extra=None, titulo='modelo'):
    columnas = FEATURES + (columnas_extra or [])
    entrenamiento = features_df[features_df['mundial'].isin([2018, 2022])]
    prueba = features_df[features_df['mundial'] == 2026]

    X_train, y_train = entrenamiento[columnas], entrenamiento['resultado_local']
    X_test, y_test = prueba[columnas], prueba['resultado_local']

    linea_base = y_train.value_counts(normalize=True).idxmax()
    exactitud_base = accuracy_score(y_test, [linea_base] * len(y_test))

    modelo = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42)
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    exactitud = accuracy_score(y_test, pred)

    print(f'\n=== {titulo} ===')
    print(f'Línea base (siempre {linea_base}): {exactitud_base:.3f}')
    print(f'Árbol de decisión: {exactitud:.3f}')

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax)
    ax.set_title(titulo)
    fig.tight_layout()
    fig.savefig(SALIDA / f'matriz_confusion_{titulo}.png')
    plt.close(fig)

    return modelo, exactitud_base, exactitud


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    d18 = pd.read_csv(DATA / 'mundial_2018_sucio.csv', dtype=str)
    d22 = pd.read_csv(DATA / 'mundial_2022_sucio.csv', dtype=str)
    d26 = pd.read_csv(DATA / 'mundial_2026_sucio.csv', dtype=str)

    for df, nombre in [(d18, '2018'), (d22, '2022'), (d26, '2026')]:
        perfil(df, nombre)

    limpio18 = limpiar_mundial(d18, 2018)
    limpio22 = limpiar_mundial(d22, 2022)
    limpio26 = limpiar_mundial(d26, 2026)
    partidos = pd.concat([limpio18, limpio22, limpio26], ignore_index=True)

    validar(partidos)

    ruta_csv = SALIDA / 'mundial_limpio.csv'
    partidos.to_csv(ruta_csv, index=False)
    print(f'\nBase limpia guardada en {ruta_csv} ({len(partidos)} partidos).')

    comparar_torneos(partidos)

    tabla = tabla_equipos(partidos)
    tabla.to_csv(SALIDA / 'tabla_equipos.csv', index=False)
    for mundial in PARTIDOS_ESPERADOS:
        mejor = tabla[tabla['mundial'] == mundial].iloc[0]
        print(f"Mejor diferencia de goles en {mundial}: {mejor['equipo']} (DG={mejor['dg']:.0f})")

    features_df = construir_variables_previas(partidos)
    entrenar_y_evaluar(features_df, titulo='sin_fuga')
    entrenar_y_evaluar(
        features_df,
        columnas_extra=['goles_local', 'goles_visitante', 'diferencia_goles'],
        titulo='con_fuga',
    )

    print('\n=== Bitácora de incidencias ===')
    for i, mensaje in enumerate(incidencias, 1):
        print(f'{i}. {mensaje}')
    (SALIDA / 'bitacora.txt').write_text('\n'.join(incidencias), encoding='utf-8')


if __name__ == '__main__':
    main()
