# Bitácora de Calidad de Datos
## Mundiales 2018, 2022 y 2026 — Fase de Grupos

---

## Perfilado inicial

| Mundial | Filas brutas | Columnas | Nulos | Duplicados detectados |
|---------|-------------|----------|-------|-----------------------|
| 2018    | ~50         | 13       | Sí    | Sí (IDs repetidos)    |
| 2022    | ~50         | 14       | Sí    | Sí                    |
| 2026    | ~74         | 14       | Sí    | Sí                    |

---

## Problemas detectados y decisiones de limpieza

| # | Incidencia | Archivo(s) | Decisión tomada | Impacto |
|---|-----------|------------|-----------------|---------|
| 1 | **3 esquemas de columnas distintos** | 2018, 2022, 2026 | Mapas de renombrado por mundial en `rename_maps` | Sin impacto en datos, solo estructura |
| 2 | **Fechas en 6+ formatos** (`2018-06-14`, `15/06/2018`, `Jun 16, 2018`, etc.) | 2018, 2022 | Parseo con fallbacks secuenciales + filtro por rango de fechas del torneo | Fecha correcta recuperada en todos los casos |
| 3 | **Serial de Excel** (ej. `46187`) | 2022 | Detectar enteros en rango 40000-50000 y sumar días al origen `1899-12-30` | Fecha correcta recuperada |
| 4 | **Fecha `"N/D"`** o vacía | 2018 | Convertir a `pd.NaT` | Fecha nula admitida; partido sigue válido |
| 5 | **5 separadores de marcador** (`-`, `–`, `—`, `:`, `x`) | 2018, 2022, 2026 | Regex que normaliza todos los separadores a `-` antes de dividir | Todos los marcadores parseados |
| 6 | **Marcador `"sin dato"`, vacío o inválido** | 2022 | Marcar como NaN; inferir goles desde columnas individuales | Goles recuperados cuando estaban disponibles individualmente |
| 7 | **Texto en columna de goles** (`"5 goles"`, `"1 goles"`) | 2018 | Regex que extrae el primer entero del string | Valor numérico correcto extraído |
| 8 | **Goles negativos** (`-1`, `-2`) | 2026 | Tratar como inválidos; intentar recuperar del marcador | Goles negativos eliminados; marcador usado si era válido |
| 9 | **Goles `"N/A"`, `"s/d"`, vacíos** | 2018, 2022 | Convertir a `np.nan`; inferir del marcador si existe | Goles recuperados cuando el marcador era legible |
| 10 | **Espacios en nombres de equipos** (`" russia "`) | 2018, 2022 | `.strip()` antes del mapeo al catálogo | Todos los equipos normalizados |
| 11 | **40+ variantes de nombre de equipo** (`USA`, `U.S.A.`, `US`, `Korea Rep.`, etc.) | 2018, 2022, 2026 | Catálogo `catalogo_equipos.csv` + función `clave_texto()` sin acentos/signos | Nombre canónico unificado por equipo |
| 12 | **Fase con 8 variantes** (`"Group Stage"`, `"Fase grupos"`, `"GS"`, etc.) | 2018, 2022, 2026 | Todo asignado como `"Fase de grupos"` | Uniformidad total |
| 13 | **Grupo en formatos mixtos** (`"group-a"`, `"A "`, `"Grupo A"`, `"s/d"`) | 2018, 2022, 2026 | Regex que extrae la letra A-L del valor | Letra de grupo normalizada |
| 14 | **Grupo vacío** en algunos partidos | 2018 | Inferir desde el catálogo de equipos por mundial (`GRUPOS_2018`) o desde filas con grupo conocido | Grupos faltantes imputados |
| 15 | **Anfitrión en formatos mixtos** (`Sí`, `si`, `TRUE`, `1`, `0`, `FALSE`) | 2018, 2022, 2026 | Función `normalizar_booleano()` con lista de valores verdaderos | Booleano consistente |
| 16 | **2026 tiene 2 columnas de anfitrión** (`host_h` y `host_a`) | 2026 | Mapear ambas a `local_es_anfitrion` y `visitante_es_anfitrion` respectivamente | Ambas columnas preservadas |
| 17 | **Filas duplicadas por `partido_id`** | 2018, 2022, 2026 | `drop_duplicates(subset=['partido_id'], keep='first')` | Duplicados eliminados sin perder datos |
| 18 | **Conflicto entre goles individuales y marcador** | 2018, 2022 | El marcador tiene prioridad cuando es parseable; si no, se usan los goles individuales; goles negativos siempre inválidos | Regla documentada y reproducible |

---

## Resultado final

| Métrica | Valor |
|---------|-------|
| Partidos 2018 | **48** ✓ |
| Partidos 2022 | **48** ✓ |
| Partidos 2026 | **72** ✓ |
| **Total** | **168** ✓ |
| Duplicados | **0** ✓ |
| Goles negativos | **0** ✓ |
| Nulos en equipos/goles/grupo | **0** ✓ |
| Marcadores inconsistentes | **0** ✓ |
| Grupos fuera de A-L | **0** ✓ |

---

## Decisión de comparación

El Mundial 2026 tiene **72 partidos** de fase de grupos frente a **48** en 2018 y 2022 (48 vs 32 equipos). Por ello, la comparación entre torneos **no debe basarse en totales brutos** sino en **tasas** (goles por partido, porcentaje de empates, porcentaje de victorias locales). Los totales crudos siempre favorecerán a 2026 por el mayor tamaño de muestra.
