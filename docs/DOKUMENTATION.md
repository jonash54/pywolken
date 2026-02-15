# pywolken Dokumentation

> **[English Documentation](DOCUMENTATION.md)**

> **Version:** 0.1.0
> **Lizenz:** MIT
> **Python:** >= 3.10
> **Repository:** [github.com/jonash54/pywolken](https://github.com/jonash54/pywolken)

pywolken ist eine reine Python-Bibliothek zur Punktwolkenverarbeitung — eine vollständige Alternative zu PDAL.
Sie liest LAS/LAZ/PLY/CSV-Dateien, wendet Filterketten über JSON-Pipelines an, erzeugt DEMs und Schummerungen, erstellt 3D-Meshes und skaliert auf Milliarden von Punkten mittels Chunk-basiertem Streaming und optionaler Dask-Parallelisierung. Keine C++-Kompilierung erforderlich.

---

## Inhaltsverzeichnis

- [Installation](#installation)
- [Schnellstart](#schnellstart)
- [Kernkonzepte](#kernkonzepte)
  - [PointCloud](#pointcloud)
  - [Bounds](#bounds)
  - [Metadata](#metadata)
  - [Dimensionen](#dimensionen)
- [I/O — Lesen und Schreiben](#io--lesen-und-schreiben)
  - [Komfortfunktionen](#komfortfunktionen)
  - [LAS/LAZ](#laslaz)
  - [PLY](#ply)
  - [CSV/TXT/XYZ](#csvtxtxyz)
  - [GeoTIFF (DEM-Writer)](#geotiff-dem-writer)
  - [I/O-Registry](#io-registry)
- [Filter](#filter)
  - [Filter-Registry](#filter-registry)
  - [filters.range](#filtersrange)
  - [filters.crop](#filterscrop)
  - [filters.merge](#filtersmerge)
  - [filters.decimation](#filtersdecimation)
  - [filters.assign](#filtersassign)
  - [filters.expression](#filtersexpression)
  - [filters.reprojection](#filtersreprojection)
  - [filters.ground](#filtersground)
  - [filters.hag](#filtershag)
  - [filters.outlier](#filtersoutlier)
  - [filters.normal](#filtersnormal)
  - [filters.voxel](#filtersvoxel)
  - [filters.cluster](#filterscluster)
  - [filters.colorize](#filterscolorize)
  - [filters.sort](#filterssort)
- [Pipeline-Engine](#pipeline-engine)
  - [JSON-Pipeline-Format](#json-pipeline-format)
  - [Pipeline-Klasse](#pipeline-klasse)
  - [StreamingPipeline](#streamingpipeline)
- [Raster-Modul](#raster-modul)
  - [DEM-Erzeugung](#dem-erzeugung)
  - [Schummerung (Hillshade)](#schummerung-hillshade)
  - [GeoTIFF-Export](#geotiff-export)
- [Mesh-Modul](#mesh-modul)
  - [Triangulierung](#triangulierung)
  - [Mesh-Export](#mesh-export)
- [Parallelverarbeitung (Dask)](#parallelverarbeitung-dask)
- [CLI-Referenz](#cli-referenz)
- [Architektur](#architektur)
- [Abhängigkeiten](#abhängigkeiten)
- [Beispiele](#beispiele)

---

## Installation

```bash
# Kern (LAS/LAZ, Filter, Pipelines)
pip install pywolken

# Mit Raster-Unterstützung (DEM, Schummerung, GeoTIFF-Export)
pip install pywolken[raster]

# Mit Dask für Parallelverarbeitung
pip install pywolken[dask]

# Alles
pip install pywolken[all]

# Entwicklung
pip install pywolken[dev]
```

### Kern-Abhängigkeiten (werden automatisch installiert)

| Paket | Zweck |
|-------|-------|
| `numpy >= 1.24` | Array-Operationen, zentrale Datenrepräsentation |
| `laspy[lazrs] >= 2.5` | LAS/LAZ-I/O (lazrs = Rust-basierter LAZ-Codec, wird als Wheel ausgeliefert) |
| `scipy >= 1.10` | KDTree-Raumindizierung, Delaunay-Triangulierung, Interpolation |
| `pyproj >= 3.5` | CRS-Definitionen und Koordinaten-Reprojektion |

### Optionale Abhängigkeiten

| Extra | Paket | Zweck |
|-------|-------|-------|
| `raster` | `rasterio >= 1.3` | GeoTIFF-Export, Raster-Kolorierung |
| `dask` | `dask[array] >= 2023.0` | Parallele/verteilte Chunk-Verarbeitung |
| `viz` | `matplotlib >= 3.7, plotly >= 5.0` | Visualisierung |
| `mesh` | `open3d >= 0.17` | Erweiterte Mesh-Verarbeitung (Poisson-Rekonstruktion) |

---

## Schnellstart

```python
import pywolken

# Beliebiges unterstütztes Format lesen (automatische Erkennung anhand der Dateiendung)
pc = pywolken.read("terrain.laz")
print(pc)
# PointCloud(45,266,951 points, dims=[X, Y, Z, Intensity, Classification, GpsTime])

# Dimensionen als NumPy-Arrays abrufen
print(pc["X"].mean())     # 399521.43
print(pc["Z"].min())      # 318.02
print(pc.bounds)          # Bounds(x=[399000.00, 400000.00], ...)
print(pc.crs)             # EPSG:25832

# Mit boolescher Maske filtern
ground = pc.mask(pc["Classification"] == 2)
print(ground.num_points)  # 23,354,571

# In beliebiges Format schreiben
pywolken.write(ground, "ground.laz")
pywolken.write(ground, "ground.ply", format="ascii")
pywolken.write(ground, "ground.csv", delimiter=";")

# JSON-Pipeline (PDAL-kompatibel)
import json
pipeline = pywolken.Pipeline(json.dumps({
    "pipeline": [
        "input.laz",
        {"type": "filters.range", "limits": "Classification[2:2]"},
        {"type": "filters.decimation", "step": 10},
        "output.las"
    ]
}))
count = pipeline.execute()
print(f"Verarbeitet: {count:,} Punkte")
```

---

## Kernkonzepte

### PointCloud

Die zentrale Datenstruktur. Speichert Punktdimensionen als Dictionary von NumPy-Arrays (spaltenorientiertes Layout). Jede Dimension (X, Y, Z, Intensity, Classification usw.) ist ein separates, zusammenhängendes Array.

**Modul:** `pywolken.core.pointcloud`

```python
from pywolken import PointCloud
import numpy as np

# Leere Punktwolke erstellen und Dimensionen hinzufügen
pc = PointCloud()
pc["X"] = np.array([1.0, 2.0, 3.0])
pc["Y"] = np.array([4.0, 5.0, 6.0])
pc["Z"] = np.array([7.0, 8.0, 9.0])

# Aus Dictionary erstellen (bevorzugt)
pc = PointCloud.from_dict({
    "X": np.array([1.0, 2.0, 3.0]),
    "Y": np.array([4.0, 5.0, 6.0]),
    "Z": np.array([7.0, 8.0, 9.0]),
    "Classification": np.array([2, 2, 6], dtype=np.uint8),
}, crs="EPSG:25832")

# Aus NumPy Structured Array erstellen
arr = np.zeros(100, dtype=[("X", "f8"), ("Y", "f8"), ("Z", "f8")])
pc = PointCloud.from_numpy(arr)
```

#### Eigenschaften

| Eigenschaft | Typ | Beschreibung |
|-------------|-----|--------------|
| `num_points` | `int` | Anzahl der Punkte |
| `dimensions` | `list[str]` | Liste der Dimensionsnamen |
| `bounds` | `Bounds` | 3D-Bounding-Box (erfordert X, Y, Z) |
| `metadata` | `Metadata` | Qülldatei-Informationen, Format usw. |
| `crs` | `str \| None` | Koordinatenreferenzsystem (EPSG oder WKT) |

#### Methoden

```python
# Array-Zugriff
pc["X"]                          # Dimension als np.ndarray abrufen
pc["X"] = new_array              # Dimension setzen
"X" in pc                        # Prüfen, ob Dimension existiert
len(pc)                          # Anzahl der Punkte

# Filtern
filtered = pc.mask(pc["Z"] > 100)          # Boolesche Maske → neü PointCloud
chunk = pc.slice(0, 1000)                  # Nach Indexbereich aufteilen

# Iteration
for chunk in pc.iter_chunks(1_000_000):    # In Chunks iterieren
    process(chunk)

# Dimensionsverwaltung
pc.add_dimension("NewDim", dtype=np.float32, fill=0.0)
pc.remove_dimension("NewDim")

# Konvertierung
structured_arr = pc.to_numpy()             # → NumPy Structured Array
dim_dict = pc.to_dict()                    # → dict[str, np.ndarray]

# Operationen
copy = pc.copy()                           # Tiefe Kopie
merged = pc.merge(other_pc)                # Zusammenführen (nur gemeinsame Dimensionen)
```

---

### Bounds

Unveränderliche Dataclass, die eine 3D-achsenausgerichtete Bounding-Box repräsentiert.

**Modul:** `pywolken.core.bounds`

```python
from pywolken import Bounds

b = Bounds(minx=0, miny=0, minz=0, maxx=100, maxy=100, maxz=50)
b = Bounds.from_arrays(x_arr, y_arr, z_arr)
b = Bounds.from_pointcloud(pc)

b.contains_point(50, 50, 25)     # Trü
b.contains_2d(50, 50)            # Trü
b.intersects(other_bounds)       # Trü/False
union = b.union(other_bounds)    # Umschliessende Box

b.width     # maxx - minx
b.height    # maxy - miny
b.depth     # maxz - minz
```

#### Attribute

| Attribut | Typ | Beschreibung |
|----------|-----|--------------|
| `minx` | `float` | Minimales X |
| `miny` | `float` | Minimales Y |
| `minz` | `float` | Minimales Z |
| `maxx` | `float` | Maximales X |
| `maxy` | `float` | Maximales Y |
| `maxz` | `float` | Maximales Z |

---

### Metadata

Dataclass mit Qüllinformationen der Punktwolke.

**Modul:** `pywolken.core.metadata`

```python
from pywolken import Metadata

m = Metadata(
    source_file="/data/terrain.laz",
    source_format="las",
    point_format_id=6,
    file_version="1.4",
    software="pywolken",
)
copy = m.copy()
```

| Attribut | Typ | Standard | Beschreibung |
|----------|-----|----------|--------------|
| `source_file` | `str \| None` | `None` | Ursprünglicher Dateipfad |
| `source_format` | `str \| None` | `None` | Format: "las", "ply", "csv" |
| `creation_date` | `datetime \| None` | `None` | Erstellungszeitstempel der Datei |
| `software` | `str` | `"pywolken"` | Erzeugende Software |
| `point_format_id` | `int \| None` | `None` | LAS-Punktformat (0-10) |
| `file_version` | `str \| None` | `None` | LAS-Version ("1.2", "1.4") |
| `extra` | `dict[str, Any]` | `{}` | Beliebige zusätzliche Metadaten |

---

### Dimensionen

Standard-LAS-Dimensionsdefinitionen und ASPRS-Klassifikationscodes.

**Modul:** `pywolken.core.dimensions`

```python
from pywolken.core.dimensions import STANDARD_DIMENSIONS, CLASSIFICATION_CODES, get_dtype

# Standard-Dimension → Datentyp-Zuordnung
STANDARD_DIMENSIONS = {
    "X": float64, "Y": float64, "Z": float64,
    "Intensity": uint16,
    "ReturnNumber": uint8, "NumberOfReturns": uint8,
    "Classification": uint8,
    "ScanAngleRank": float32,
    "UserData": uint8, "PointSourceId": uint16,
    "GpsTime": float64,
    "Red": uint16, "Green": uint16, "Blü": uint16, "NIR": uint16,
}

# ASPRS-Klassifikationscodes
CLASSIFICATION_CODES = {
    0: "Created, Never Classified",
    1: "Unclassified",
    2: "Ground",
    3: "Low Vegetation",
    4: "Medium Vegetation",
    5: "High Vegetation",
    6: "Building",
    7: "Low Point (Noise)",
    9: "Water",
    10: "Rail",
    11: "Road Surface",
    17: "Bridge Deck",
    18: "High Noise",
}

dtype = get_dtype("Classification")  # uint8
dtype = get_dtype("CustomDim")       # float64 (Standard)
```

---

## I/O — Lesen und Schreiben

### Komfortfunktionen

**Modul:** `pywolken.io.registry`
**Auch exportiert von:** `pywolken` (oberste Ebene)

```python
import pywolken

# Format automatisch anhand der Dateiendung erkennen
pc = pywolken.read("file.laz")
pc = pywolken.read("file.ply")
pc = pywolken.read("file.csv", delimiter=";")

# Schreiben (Format anhand der Endung)
pywolken.write(pc, "output.las")
pywolken.write(pc, "output.ply", format="ascii")
pywolken.write(pc, "output.csv", precision=3)

# Chunk-weises Lesen für grosse Dateien
for chunk in pywolken.read_chunked("huge.laz", chunk_size=2_000_000):
    process(chunk)
```

#### `read(path, **options) -> PointCloud`

Liest eine Punktwolken-Datei. Format wird automatisch anhand der Dateiendung erkannt.

| Parameter | Typ | Beschreibung |
|-----------|-----|--------------|
| `path` | `str` | Dateipfad (.las, .laz, .ply, .csv, .txt, .xyz) |
| `**options` | | Formatspezifische Optionen (siehe unten je Format) |

#### `write(pc, path, **options) -> int`

Schreibt eine Punktwolke. Gibt die Anzahl geschriebener Punkte zurück.

| Parameter | Typ | Beschreibung |
|-----------|-----|--------------|
| `pc` | `PointCloud` | Zu schreibende Punktwolke |
| `path` | `str` | Ausgabepfad (Format anhand der Endung) |
| `**options` | | Formatspezifische Optionen |

#### `read_chunked(path, chunk_size=1_000_000, **options) -> Iterator[PointCloud]`

Liest eine Datei in Chunks per Streaming. LAS/LAZ nutzt natives Chunk-Lesen für echtes Streaming.

---

### LAS/LAZ

**Reader-Typ:** `readers.las`
**Writer-Typ:** `writers.las`
**Dateiendungen:** `.las`, `.laz`

Verwendet `laspy[lazrs]` — lazrs bietet Rust-basierte LAZ-Kompression als vorgefertigtes Wheel.

```python
from pywolken.io.las import LasReader, LasWriter

# Lesen
reader = LasReader()
pc = reader.read("terrain.laz")

# Chunk-weises Lesen (speichereffizient)
for chunk in reader.read_chunked("huge.laz", chunk_size=1_000_000):
    process(chunk)

# Schreiben
writer = LasWriter()
writer.write(pc, "output.las")
writer.write(pc, "output.laz")  # komprimiert
writer.write(pc, "output.las", point_format_id=6, file_version="1.4")
```

#### Writer-Optionen

| Option | Typ | Standard | Beschreibung |
|--------|-----|----------|--------------|
| `point_format_id` | `int` | automatisch | LAS-Punktformat (0-10) |
| `file_version` | `str` | automatisch | "1.2" für Formate 0-5, "1.4" für Formate 6-10 |

**Automatische Erkennung:**
- Format 8 → hat NIR
- Format 7 → hat RGB + GpsTime
- Format 2 → hat RGB
- Format 6 → hat GpsTime
- Format 0 → Basis

**Dimensionszuordnung (laspy Kleinbuchstaben → pywolken Grossbuchstaben):**

| laspy | pywolken |
|-------|----------|
| `x`, `y`, `z` | `X`, `Y`, `Z` |
| `intensity` | `Intensity` |
| `return_number` | `ReturnNumber` |
| `number_of_returns` | `NumberOfReturns` |
| `classification` | `Classification` |
| `scan_angle_rank` / `scan_angle` | `ScanAngleRank` |
| `user_data` | `UserData` |
| `point_source_id` | `PointSourceId` |
| `gps_time` | `GpsTime` |
| `red`, `green`, `blü` | `Red`, `Green`, `Blü` |
| `nir` | `NIR` |

Zusätzliche Dimensionen, die nicht in dieser Zuordnung enthalten sind, werden mit ihren Originalnamen beibehalten.

---

### PLY

**Reader-Typ:** `readers.ply`
**Writer-Typ:** `writers.ply`
**Dateiendungen:** `.ply`

Stanford-PLY-Format. Unterstützt ASCII, Binary Little-Endian und Binary Big-Endian.

```python
from pywolken.io.ply import PlyReader, PlyWriter

pc = PlyReader().read("model.ply")
PlyWriter().write(pc, "output.ply", format="ascii")
PlyWriter().write(pc, "output.ply", format="binary_little_endian")
```

#### Writer-Optionen

| Option | Typ | Standard | Beschreibung |
|--------|-----|----------|--------------|
| `format` | `str` | `"binary_little_endian"` | `"ascii"`, `"binary_little_endian"` oder `"binary_big_endian"` |

**PLY ↔ pywolken Dimensionszuordnung:**

| PLY | pywolken |
|-----|----------|
| `x`, `y`, `z` | `X`, `Y`, `Z` |
| `nx`, `ny`, `nz` | `NormalX`, `NormalY`, `NormalZ` |
| `red`, `green`, `blü`, `alpha` | `Red`, `Green`, `Blü`, `Alpha` |
| `intensity` | `Intensity` |
| `classification` | `Classification` |
| `gps_time` | `GpsTime` |

**Unterstützte PLY-Datentypen:** `char`, `uchar`, `short`, `ushort`, `int`, `uint`, `float`, `double`, `int8`-`int64`, `uint8`-`uint64`, `float32`, `float64`

---

### CSV/TXT/XYZ

**Reader-Typ:** `readers.csv`
**Writer-Typ:** `writers.csv`
**Dateiendungen:** `.csv`, `.txt`, `.xyz`

Trennzeichen-basiertes Textformat. Erkennt automatisch Trennzeichen aus `,`, `;`, `\t`, ` `.

```python
from pywolken.io.csv import CsvReader, CsvWriter

# Trennzeichen und Kopfzeile automatisch erkennen
pc = CsvReader().read("points.csv")

# Explizite Optionen
pc = CsvReader().read("points.txt", delimiter="\t", skip=2)

# Keine Kopfzeile — Spaltennamen angeben
pc = CsvReader().read("points.xyz", header="X,Y,Z,Intensity")

# Schreiben
CsvWriter().write(pc, "output.csv")
CsvWriter().write(pc, "output.csv", delimiter=";", precision=3, header=Trü)
```

#### Reader-Optionen

| Option | Typ | Standard | Beschreibung |
|--------|-----|----------|--------------|
| `delimiter` | `str` | automatisch | Feldtrenner |
| `header` | `str` | `None` | Kommagetrennte Spaltennamen (falls Datei keine Kopfzeile hat) |
| `skip` | `int` | `0` | Zeilen, die vor den Daten übersprungen werden |

#### Writer-Optionen

| Option | Typ | Standard | Beschreibung |
|--------|-----|----------|--------------|
| `delimiter` | `str` | `","` | Feldtrenner |
| `header` | `bool` | `Trü` | Kopfzeile schreiben |
| `precision` | `int` | `6` | Nachkommastellen für Gleitkommazahlen |

---

### GeoTIFF (DEM-Writer)

**Writer-Typ:** `writers.gdal`
**Dateiendungen:** `.tif`, `.tiff`

Erzeugt ein DEM aus einer Punktwolke und schreibt es als GeoTIFF. Erfordert `rasterio`.

```python
from pywolken.io.gdal import GdalWriter

writer = GdalWriter()
writer.write(pc, "dem.tif", resolution=0.5, output_type="idw")
```

#### Writer-Optionen

| Option | Typ | Standard | Beschreibung |
|--------|-----|----------|--------------|
| `resolution` | `float` | **erforderlich** | Rasterzellengrösse in CRS-Einheiten |
| `output_type` | `str` | `"idw"` | Interpolation: `"idw"`, `"mean"`, `"nearest"`, `"tin"` |
| `window_size` | `int` | `6` | IDW-Suchfenster (Zellen) |
| `power` | `float` | `2.0` | IDW-Distanzpotenz |
| `nodata` | `float` | `-9999.0` | Nodata-Füllwert |
| `gdalopts` | `str` | `""` | GDAL-Erstellungsoptionen: `"COMPRESS=LZW,TILED=YES"` |

---

### I/O-Registry

Automatische Formaterkennung und Plugin-Registrierung.

```python
from pywolken.io.registry import io_registry

# Eigenen Reader/Writer registrieren
io_registry.register_reader(MyCustomReader)
io_registry.register_writer(MyCustomWriter)

# Reader/Writer anhand der Dateiendung abrufen
reader = io_registry.get_reader("file.laz")     # → LasReader
writer = io_registry.get_writer("output.ply")    # → PlyWriter

# Anhand des Pipeline-Typnamens abrufen
reader = io_registry.get_reader_by_type("readers.las")
writer = io_registry.get_writer_by_type("writers.csv")
```

---

## Filter

Alle Filter folgen demselben Muster: Mit Optionen instanziieren, `filter(pc)` aufrufen, um eine neü PointCloud zu erhalten. Filter sind unveränderlich — sie modifizieren niemals die Eingabe.

```python
from pywolken.filters.range import RangeFilter

f = RangeFilter(limits="Classification[2:2]")
result = f.filter(pc)
```

### Filter-Registry

Alle 15 integrierten Filter sind registriert und können über den Typnamen erstellt werden:

```python
from pywolken.filters.registry import get_filter, filter_registry

# über Typnamen erstellen
f = get_filter("filters.range", limits="Classification[2:2]")
result = f.filter(pc)

# Alle verfügbaren Filter auflisten
print(filter_registry.available)
# ['filters.range', 'filters.crop', 'filters.merge', 'filters.decimation',
#  'filters.assign', 'filters.expression', 'filters.reprojection',
#  'filters.ground', 'filters.hag', 'filters.outlier', 'filters.normal',
#  'filters.voxel', 'filters.cluster', 'filters.colorize', 'filters.sort']
```

---

### filters.range

Filtert Punkte nach Dimensionswertebereichen. PDAL-kompatible Syntax.

**Typname:** `filters.range`

```python
from pywolken.filters.range import RangeFilter

f = RangeFilter(limits="Classification[2:2]")           # Exakt Klasse 2
f = RangeFilter(limits="Z[100:500]")                     # Z zwischen 100-500
f = RangeFilter(limits="Z[100:]")                        # Z >= 100
f = RangeFilter(limits="Z[:500]")                        # Z <= 500
f = RangeFilter(limits="Classification![7:7]")           # Rauschen ausschliessen (Klasse 7)
f = RangeFilter(limits="Classification[2:2],Z[100:]")    # Mehrfach (UND)
result = f.filter(pc)
```

| Option | Typ | Erforderlich | Beschreibung |
|--------|-----|--------------|--------------|
| `limits` | `str` | ja | PDAL-Bereichsausdruck |

**Syntax:** `Dimensionsname[min:max]` — eckige Klammern inklusive. Mit `!` vorangestellt zum Negieren. Mehrere Bedingungen kommagetrennt (UND-Logik).

---

### filters.crop

Räumlicher Bounding-Box-Zuschnitt.

**Typname:** `filters.crop`

```python
from pywolken.filters.crop import CropFilter

# PDAL-Bounds-String
f = CropFilter(bounds="([399000, 400000], [5600000, 5601000])")

# 3D-Bounds
f = CropFilter(bounds="([399000, 400000], [5600000, 5601000], [100, 500])")

# Explizite Werte
f = CropFilter(minx=399000, maxx=400000, miny=5600000, maxy=5601000)

result = f.filter(pc)
```

| Option | Typ | Erforderlich | Beschreibung |
|--------|-----|--------------|--------------|
| `bounds` | `str` | eines von | PDAL-Syntax: `"([xmin,xmax],[ymin,ymax])"` |
| `minx`, `maxx`, `miny`, `maxy` | `float` | eines von | Explizite 2D-Bounds |
| `minz`, `maxz` | `float` | nein | Optionale Z-Bounds |

---

### filters.merge

Mehrere Punktwolken zusammenführen. Nur gemeinsame Dimensionen.

**Typname:** `filters.merge`

```python
from pywolken.filters.merge import MergeFilter

f = MergeFilter()
f.add_input(other_pc1)
f.add_input(other_pc2)
result = f.filter(pc)  # Führt pc + other_pc1 + other_pc2 zusammen
```

---

### filters.decimation

Punktanzahl durch Unterabtastung reduzieren.

**Typname:** `filters.decimation`

```python
from pywolken.filters.decimation import DecimationFilter

f = DecimationFilter(step=10)                       # Jeden 10. Punkt
f = DecimationFilter(fraction=0.1)                  # Zufällige 10%
f = DecimationFilter(count=100000)                  # Exakt 100k Punkte
f = DecimationFilter(fraction=0.5, seed=42)         # Reproduzierbar zufällig
result = f.filter(pc)
```

| Option | Typ | Erforderlich | Beschreibung |
|--------|-----|--------------|--------------|
| `step` | `int` | eines von | Jeden N-ten Punkt behalten |
| `fraction` | `float` | eines von | Diesen Anteil behalten (0.0, 1.0] |
| `count` | `int` | eines von | Exakt diese Anzahl behalten |
| `seed` | `int` | nein | Zufalls-Seed für Reproduzierbarkeit |

---

### filters.assign

Dimensionswerte auf Konstanten setzen.

**Typname:** `filters.assign`

```python
from pywolken.filters.assign import AssignFilter

f = AssignFilter(assignment="Classification=2,Intensity=0")
f = AssignFilter(valü={"Classification": 2, "Intensity": 0})
result = f.filter(pc)
```

| Option | Typ | Erforderlich | Beschreibung |
|--------|-----|--------------|--------------|
| `assignment` | `str` | eines von | `"Dim=Wert,Dim=Wert"` |
| `valü` | `dict` | eines von | `{"Dim": Wert}` |

Erstellt neü Dimensionen, falls diese nicht existieren.

---

### filters.expression

Punkte durch boolesche Ausdrücke filtern.

**Typname:** `filters.expression`

```python
from pywolken.filters.expression import ExpressionFilter

f = ExpressionFilter(expression="Classification == 2")
f = ExpressionFilter(expression="Z > 100 AND Z < 500")
f = ExpressionFilter(expression="Classification == 2 OR Classification == 6")
f = ExpressionFilter(where="Intensity >= 500")  # 'where'-Alias
result = f.filter(pc)
```

| Option | Typ | Erforderlich | Beschreibung |
|--------|-----|--------------|--------------|
| `expression` | `str` | eines von | Boolescher Ausdruck |
| `where` | `str` | eines von | Alias für expression |

**Operatoren:** `==`, `!=`, `>`, `<`, `>=`, `<=`
**Kombinatoren:** `AND`, `OR`

---

### filters.reprojection

Koordinaten zwischen CRS transformieren. Erfordert `pyproj`.

**Typname:** `filters.reprojection`

```python
from pywolken.filters.reprojection import ReprojectionFilter

f = ReprojectionFilter(out_srs="EPSG:4326")                       # Automatisches Qüll-CRS
f = ReprojectionFilter(in_srs="EPSG:25832", out_srs="EPSG:4326")  # Explizite Qülle
result = f.filter(pc)
```

| Option | Typ | Erforderlich | Beschreibung |
|--------|-----|--------------|--------------|
| `out_srs` | `str` | ja | Ziel-CRS (z.B. `"EPSG:4326"`) |
| `in_srs` | `str` | nein | Qüll-CRS (verwendet `pc.crs`, falls nicht angegeben) |

Aktualisiert `pc.crs` im Ergebnis.

---

### filters.ground

SMRF (Simple Morphological Filter) Bodenklassifikation. Setzt `Classification=2` für Bodenpunkte, `Classification=1` für Nicht-Bodenpunkte.

**Typname:** `filters.ground`

```python
from pywolken.filters.ground import GroundFilter

f = GroundFilter()
f = GroundFilter(cell_size=1.0, slope=0.15, window_max=18.0, threshold=0.5)
result = f.filter(pc)
ground_only = result.mask(result["Classification"] == 2)
```

| Option | Typ | Standard | Beschreibung |
|--------|-----|----------|--------------|
| `cell_size` | `float` | `1.0` | Anfängliche Mindest-Rasterzellengrösse der Oberfläche |
| `slope` | `float` | `0.15` | Maximale Geländeneigung |
| `window_max` | `float` | `18.0` | Maximale morphologische Fenstergrösse |
| `threshold` | `float` | `0.5` | Schwellenwert für Höhenunterschied |
| `scalar` | `float` | `1.25` | Neigungsskalierungsfaktor |

---

### filters.hag

Height Above Ground — berechnet die Höhe jedes Punktes relativ zur Bodenoberfläche.

**Typname:** `filters.hag`

**Voraussetzung:** Die Punktwolke muss eine `Classification`-Dimension mit Bodenpunkten (`Classification == 2`) besitzen.

```python
from pywolken.filters.hag import HagFilter

f = HagFilter(neighbors=3)
result = f.filter(pc)
print(result["HeightAboveGround"])  # Neü Dimension
```

| Option | Typ | Standard | Beschreibung |
|--------|-----|----------|--------------|
| `neighbors` | `int` | `3` | Bodennachbarn für gemittelte Interpolation |

**Ausgabe:** Fügt die Dimension `HeightAboveGround` hinzu (float64).

---

### filters.outlier

Statistische oder radiusbasierte Ausreisserentfernung mittels KDTree.

**Typname:** `filters.outlier`

```python
from pywolken.filters.outlier import OutlierFilter

# Statistisch: entfernen, wenn mean_dist > Mittelwert + multiplier * Standardabweichung
f = OutlierFilter(method="statistical", mean_k=8, multiplier=2.0)

# Radius: entfernen, wenn weniger als min_k Nachbarn innerhalb des Radius
f = OutlierFilter(method="radius", radius=5.0, min_k=3)

result = f.filter(pc)  # Gibt PointCloud ohne Ausreisser zurück
```

| Option | Typ | Standard | Beschreibung |
|--------|-----|----------|--------------|
| `method` | `str` | `"statistical"` | `"statistical"` oder `"radius"` |
| `mean_k` | `int` | `8` | (statistisch) Zu analysierende Nachbarn |
| `multiplier` | `float` | `2.0` | (statistisch) Standardabweichungs-Multiplikator |
| `radius` | `float` | — | (Radius) Suchradius |
| `min_k` | `int` | `2` | (Radius) Mindestanzahl benötigter Nachbarn |

---

### filters.normal

Schätzung von Oberflächennormalen mittels PCA auf KDTree-Nachbarschaften.

**Typname:** `filters.normal`

```python
from pywolken.filters.normal import NormalFilter

f = NormalFilter(k=8)
result = f.filter(pc)
print(result["NormalX"], result["NormalY"], result["NormalZ"])
```

| Option | Typ | Standard | Beschreibung |
|--------|-----|----------|--------------|
| `k` | `int` | `8` | Nachbarn für PCA-Normalenschätzung |

**Ausgabe:** Fügt die Dimensionen `NormalX`, `NormalY`, `NormalZ` hinzu (float64). Normalen sind nach oben orientiert (NormalZ > 0).

---

### filters.voxel

Voxelraster-Downsampling. Ersetzt alle Punkte innerhalb jeder Voxelzelle durch einen einzelnen Schwerpunkt.

**Typname:** `filters.voxel`

```python
from pywolken.filters.voxel import VoxelFilter

f = VoxelFilter(cell_size=0.5)
result = f.filter(pc)
```

| Option | Typ | Standard | Beschreibung |
|--------|-----|----------|--------------|
| `cell_size` | `float` | `1.0` | Voxel-Kantenlänge in CRS-Einheiten |
| `cell` | `float` | — | Alias für `cell_size` |

X, Y, Z werden gemittelt (Schwerpunkt). Andere Dimensionen übernehmen den Wert des ersten Punktes.

---

### filters.cluster

DBSCAN-basiertes Punktwolken-Clustering.

**Typname:** `filters.cluster`

```python
from pywolken.filters.cluster import ClusterFilter

f = ClusterFilter(tolerance=5.0, min_points=10)
result = f.filter(pc)
cluster_ids = result["ClusterID"]  # 0 = Rauschen, 1+ = Cluster-ID
```

| Option | Typ | Standard | Beschreibung |
|--------|-----|----------|--------------|
| `tolerance` | `float` | `1.0` | Maximaler Abstand zwischen Cluster-Nachbarn |
| `min_points` | `int` | `10` | Mindestpunkte pro Cluster |
| `is3d` | `bool` | `Trü` | 3D-Distanz verwenden (False = nur 2D XY) |

**Ausgabe:** Fügt die Dimension `ClusterID` hinzu (int64). `0` = Rauschen/nicht zugewiesen.

---

### filters.colorize

RGB-Werte aus einem Raster-Overlay zuweisen (z.B. Orthofoto). Erfordert `rasterio`.

**Typname:** `filters.colorize`

```python
from pywolken.filters.colorize import ColorizeFilter

f = ColorizeFilter(raster="/data/orthophoto.tif")
result = f.filter(pc)
```

| Option | Typ | Erforderlich | Beschreibung |
|--------|-----|--------------|--------------|
| `raster` | `str` | ja | Pfad zur Rasterdatei (GeoTIFF usw.) |

**Ausgabe:** Fügt `Red`, `Green`, `Blü`-Dimensionen hinzu bzw. aktualisiert sie (uint16). 8-Bit-Raster werden auf 16-Bit skaliert (x257).

---

### filters.sort

Räumliche Sortierung für verbesserte Cache-Lokalität.

**Typname:** `filters.sort`

```python
from pywolken.filters.sort import SortFilter

f = SortFilter(order="morton")   # Z-Ordnungskurve (Standard)
f = SortFilter(order="xyz")      # Lexikographisch X, dann Y, dann Z
result = f.filter(pc)
```

| Option | Typ | Standard | Beschreibung |
|--------|-----|----------|--------------|
| `order` | `str` | `"morton"` | `"morton"` (Z-Ordnungskurve) oder `"xyz"` (lexikographisch) |

Morton-Sortierung bildet 2D-XY-Koordinaten mit 16-Bit-Präzision auf eine 1D-Z-Ordnungskurve ab, sodass räumlich benachbarte Punkte auch im Array nahe beieinander liegen.

---

## Pipeline-Engine

### JSON-Pipeline-Format

pywolken verwendet PDAL-kompatible JSON-Pipelines. Eine Pipeline ist eine Liste von Stufen: Reader, Filter und Writer.

```json
{
  "pipeline": [
    "input.laz",
    {"type": "filters.range", "limits": "Classification[2:2]"},
    {"type": "filters.decimation", "step": 10},
    {"type": "filters.voxel", "cell_size": 0.5},
    "output.las"
  ]
}
```

#### Stufentypen

**Einfache Strings** werden automatisch als Reader (erster) oder Writer (nachfolgende) erkannt:
```json
["input.laz", "output.las"]
```

**Dict-Stufen** mit explizitem `type`:
```json
{"type": "readers.las", "filename": "input.laz"}
{"type": "filters.range", "limits": "Z[100:]"}
{"type": "writers.las", "filename": "output.las"}
```

#### Unterstützte Reader-Typen
`readers.las`, `readers.ply`, `readers.csv`

#### Unterstützte Writer-Typen
`writers.las`, `writers.ply`, `writers.csv`, `writers.gdal`

#### Unterstützte Filter-Typen
`filters.range`, `filters.crop`, `filters.merge`, `filters.decimation`, `filters.assign`, `filters.expression`, `filters.reprojection`, `filters.ground`, `filters.hag`, `filters.outlier`, `filters.normal`, `filters.voxel`, `filters.cluster`, `filters.colorize`, `filters.sort`

---

### Pipeline-Klasse

**Modul:** `pywolken.pipeline.pipeline`

```python
from pywolken import Pipeline
import json

# Aus JSON-String
p = Pipeline(json.dumps({
    "pipeline": [
        "input.laz",
        {"type": "filters.ground"},
        {"type": "filters.range", "limits": "Classification[2:2]"},
        "ground.laz"
    ]
}))

# Validieren
errors = p.validate()  # [] falls gültig

# Ausführen
count = p.execute()     # Gibt Gesamtzahl verarbeiteter Punkte zurück

# Auf Ergebnisse zugreifen
pc = p.result           # PointCloud der letzten Ausführung
arrays = p.arrays       # dict[str, np.ndarray]
meta = p.metadata       # dict mit Qüllinformationen

# Zurück nach JSON serialisieren
json_str = p.to_json()
```

#### Methoden

| Methode | Rückgabe | Beschreibung |
|---------|-----------|--------------|
| `validate()` | `list[str]` | Fehlermeldungen (leer = gültig) |
| `execute()` | `int` | Pipeline ausführen, Punktanzahl zurückgeben |
| `to_json()` | `str` | Als JSON-String serialisieren |

#### Eigenschaften

| Eigenschaft | Typ | Beschreibung |
|-------------|-----|--------------|
| `result` | `PointCloud \| None` | Ergebnis der letzten Ausführung |
| `arrays` | `dict[str, np.ndarray] \| None` | Punktdaten der letzten Ausführung |
| `metadata` | `dict[str, Any]` | Metadaten der letzten Ausführung |

---

### StreamingPipeline

Speicherbegrenzte Pipeline, die Daten in Chunks verarbeitet. Liest Chunks, wendet Filter pro Chunk an und schreibt inkrementell. Geeignet für Dateien, die grösser als der verfügbare Arbeitsspeicher sind.

**Modul:** `pywolken.pipeline.streaming`

```python
from pywolken import StreamingPipeline
from pywolken.io.las import LasReader, LasWriter
from pywolken.filters.range import RangeFilter
from pywolken.filters.expression import ExpressionFilter

sp = StreamingPipeline(
    reader=LasReader(),
    input_path="huge.laz",
    writer=LasWriter(),
    output_path="filtered.laz",
    filters=[
        RangeFilter(limits="Classification[2:2]"),
        ExpressionFilter(expression="Z > 100"),
    ],
    chunk_size=2_000_000,
)

total = sp.execute()
print(f"Verarbeitet: {sp.total_read:,} → {sp.total_written:,} Punkte")

# Oder ohne Schreiben iterieren
for chunk in sp.iter_results():
    process(chunk)
```

#### Konstruktor-Parameter

| Parameter | Typ | Standard | Beschreibung |
|-----------|-----|----------|--------------|
| `reader` | `Reader` | **erforderlich** | Reader-Instanz |
| `input_path` | `str` | **erforderlich** | Eingabedateipfad |
| `writer` | `Writer \| None` | `None` | Writer-Instanz |
| `output_path` | `str \| None` | `None` | Ausgabedateipfad |
| `filters` | `list[Filter] \| None` | `None` | Filterkette |
| `chunk_size` | `int` | `1_000_000` | Punkte pro Chunk |
| `reader_options` | `dict` | `{}` | Zusätzliche Reader-Optionen |
| `writer_options` | `dict` | `{}` | Zusätzliche Writer-Optionen |

#### Streaming-kompatible Filter

Diese Filter funktionieren korrekt im Chunk-weisen Modus:
- `filters.range`, `filters.crop`, `filters.expression`, `filters.assign`
- `filters.decimation`, `filters.voxel`, `filters.sort`

Diese Filter benötigen globalen Kontext und sollten **nicht** im Streaming-Modus verwendet werden:
- `filters.ground`, `filters.cluster`, `filters.normal`, `filters.outlier`, `filters.hag`

---

## Raster-Modul

### DEM-Erzeugung

Digitale Höhenmodelle aus Punktwolken erzeugen.

**Modul:** `pywolken.raster.dem`

```python
from pywolken.raster.dem import create_dem

raster, transform = create_dem(
    pc,
    resolution=0.5,       # 0.5m Rasterzellen
    method="idw",          # Interpolationsmethode
    power=2.0,             # IDW-Distanzpotenz
)

print(raster.shape)        # (2000, 2000)
print(raster.dtype)        # float32
print(transform)           # {'xmin': ..., 'ymax': ..., 'resolution': 0.5, ...}
```

#### `create_dem(pc, resolution, method="idw", bounds=None, nodata=-9999.0, power=2.0, radius=None, window_size=6)`

| Parameter | Typ | Standard | Beschreibung |
|-----------|-----|----------|--------------|
| `pc` | `PointCloud` | **erforderlich** | Punktwolke mit X, Y, Z |
| `resolution` | `float` | **erforderlich** | Rasterzellengrösse in CRS-Einheiten |
| `method` | `str` | `"idw"` | `"idw"`, `"mean"`, `"nearest"` oder `"tin"` |
| `bounds` | `tuple` | `None` | `(xmin, ymin, xmax, ymax)` oder automatisch aus Daten |
| `nodata` | `float` | `-9999.0` | Nodata-Füllwert |
| `power` | `float` | `2.0` | IDW-Distanzpotenz |
| `radius` | `float` | `None` | Suchradius (automatisch falls None) |
| `window_size` | `int` | `6` | IDW-Suchfenster in Rasterzellen |

**Rückgabe:** `(raster, transform_info)` wobei:
- `raster`: 2D `float32`-Array (Zeilen x Spalten, Ursprung oben links)
- `transform_info`: Dict mit Schlüsseln `xmin`, `ymax`, `resolution`, `nrows`, `ncols`, `crs`

#### Interpolationsmethoden

| Methode | Beschreibung | Geschwindigkeit | Qualität |
|---------|--------------|-----------------|-----------|
| `"idw"` | Inverse Distanzgewichtung (Batch-KDTree) | Mittel | Gut |
| `"mean"` | Rasterzellen-Mittelwert (np.add.at Binning) | Schnell | Geringer |
| `"nearest"` | Nächster Nachbar (KDTree) | Schnell | Blockartig |
| `"tin"` | Delaunay-TIN + lineare Interpolation | Langsam | Beste |

---

### Schummerung (Hillshade)

Schummerungsberechnung aus einem DEM nach der Horn-Methode.

**Modul:** `pywolken.raster.hillshade`

```python
from pywolken.raster.hillshade import hillshade, multi_directional_hillshade

# Einrichtungs-Schummerung
hs = hillshade(
    dem=raster,
    resolution=0.5,
    azimuth=315.0,     # Licht von NW
    altitude=45.0,     # 45 Grad über dem Horizont
    z_factor=2.0,      # Vertikale überhöhung
)
# hs.shape == dem.shape, hs.dtype == uint8

# Multirichtungs-Schummerung (gemischt aus 8 Richtungen)
hs_multi = multi_directional_hillshade(
    dem=raster,
    resolution=0.5,
    z_factor=2.0,
)
```

#### `hillshade(dem, resolution, azimuth=315.0, altitude=45.0, z_factor=1.0, nodata=-9999.0)`

| Parameter | Typ | Standard | Beschreibung |
|-----------|-----|----------|--------------|
| `dem` | `np.ndarray` | **erforderlich** | 2D-Float-Array |
| `resolution` | `float` | **erforderlich** | Zellengrösse in CRS-Einheiten |
| `azimuth` | `float` | `315.0` | Lichtrichtung (0=N, 90=O, 180=S, 270=W) |
| `altitude` | `float` | `45.0` | Lichthöhe (0=Horizont, 90=Zenit) |
| `z_factor` | `float` | `1.0` | Vertikale überhöhung |
| `nodata` | `float` | `-9999.0` | Nodata-Wert im DEM |

**Rückgabe:** 2D `uint8`-Array (0-255 Schummerung).

#### `multi_directional_hillshade(dem, resolution, altitude=45.0, z_factor=1.0, nodata=-9999.0, directions=None, weights=None)`

Mischt Schummerungen aus mehreren Azimutrichtungen (Standard: 8 Haupt-/Nebenrichtungen).

---

### GeoTIFF-Export

Raster-Arrays als GeoTIFF schreiben. Erfordert `rasterio`.

**Modul:** `pywolken.raster.export`

```python
from pywolken.raster.export import write_geotiff

write_geotiff(
    array=hillshade_array,
    path="hillshade.tif",
    transform_info=transform,     # von create_dem()
    nodata=None,                  # Kein Nodata für Schummerung
    compress="lzw",
    dtype="uint8",
)
```

#### `write_geotiff(array, path, transform_info, nodata=-9999.0, compress="lzw", dtype=None)`

| Parameter | Typ | Standard | Beschreibung |
|-----------|-----|----------|--------------|
| `array` | `np.ndarray` | **erforderlich** | 2D-Raster-Array |
| `path` | `str` | **erforderlich** | Ausgabedateipfad |
| `transform_info` | `dict` | **erforderlich** | Von `create_dem()` — benötigt `xmin`, `ymax`, `resolution`, `nrows`, `ncols`, `crs` |
| `nodata` | `float \| None` | `-9999.0` | Nodata-Wert (None zum Weglassen) |
| `compress` | `str` | `"lzw"` | `"lzw"`, `"deflate"`, `"none"` |
| `dtype` | `str \| None` | `None` | Ausgabe-Datentyp (automatisch aus Array) |

---

## Mesh-Modul

### Triangulierung

2.5D-Delaunay-Triangulierung — Projektion auf XY, Triangulierung, Verwendung von Z für 3D-Vertices.

**Modul:** `pywolken.mesh.triangulate`

```python
from pywolken.mesh import triangulate_2d, Mesh

# Mesh aus Punktwolke erstellen
mesh = triangulate_2d(pc)
mesh = triangulate_2d(pc, max_edge_length=10.0)  # Dreiecke mit langen Kanten entfernen

print(mesh)  # Mesh(45,000 vertices, 89,000 faces)
```

#### `triangulate_2d(pc, max_edge_length=None) -> Mesh`

| Parameter | Typ | Standard | Beschreibung |
|-----------|-----|----------|--------------|
| `pc` | `PointCloud` | **erforderlich** | Punktwolke mit X, Y, Z |
| `max_edge_length` | `float \| None` | `None` | Dreiecke mit längeren Kanten entfernen |

Extrahiert automatisch Vertex-Farben aus Red/Green/Blü-Dimensionen, falls vorhanden.

### Mesh-Klasse

```python
mesh.num_vertices    # int
mesh.num_faces       # int
mesh.vertices        # (N, 3) float64 Array
mesh.faces           # (M, 3) int64 Array
mesh.vertex_colors   # (N, 3) uint8 Array oder None
```

### Mesh-Export

```python
# In bestimmtes Format schreiben
mesh.write_obj("output.obj")    # Wavefront OBJ
mesh.write_stl("output.stl")    # Binär-STL
mesh.write_ply("output.ply")    # ASCII PLY (mit Farben falls verfügbar)

# Automatische Erkennung anhand der Endung
mesh.write("output.obj")

# Dezimieren (Dreiecksanzahl reduzieren)
decimated = mesh.decimate(target_faces=10000)
decimated.write("simplified.obj")
```

---

## Parallelverarbeitung (Dask)

Optionale Dask-Integration für parallele Chunk-Verarbeitung. Erfordert `pip install pywolken[dask]`.

**Modul:** `pywolken.parallel`

```python
from pywolken.parallel import parallel_read, parallel_filter, parallel_apply

# Mehrere Dateien parallel lesen
pc = parallel_read(["tile1.laz", "tile2.laz", "tile3.laz"])

# Filter parallel in Chunks anwenden
from pywolken.filters.range import RangeFilter
from pywolken.filters.expression import ExpressionFilter

result = parallel_filter(
    pc,
    filters=[
        RangeFilter(limits="Classification[2:2]"),
        ExpressionFilter(expression="Z > 100"),
    ],
    n_chunks=8,
)

# Benutzerdefinierte Funktion parallel anwenden
result = parallel_apply(pc, lambda chunk: chunk.mask(chunk["Z"] > 100), n_chunks=4)
```

#### `parallel_read(paths, reader=None, **options) -> PointCloud`

Mehrere Dateien parallel lesen und zusammenführen.

#### `parallel_filter(pc, filters, n_chunks=4) -> PointCloud`

Punktwolke in Chunks aufteilen, Filter parallel anwenden, Ergebnisse zusammenführen.

**Nur für punktweise Filter** (range, expression, assign, decimation) — **NICHT** für räumliche Filter (ground, cluster, normal, outlier).

#### `parallel_apply(pc, func, n_chunks=4) -> PointCloud`

Eine benutzerdefinierte `PointCloud -> PointCloud`-Funktion parallel auf Chunks anwenden.

---

## CLI-Referenz

Der Befehl `pywolken` wird als Konsolen-Skript-Einstiegspunkt installiert.

### `pywolken info`

Informationen über eine Punktwolken-Datei anzeigen.

```bash
pywolken info terrain.laz
```

Ausgabe:
```
File: terrain.laz
Size: 289.3 MB
Points: 45,266,951
Dimensions: X, Y, Z, Intensity, ReturnNumber, ...
CRS: EPSG:25832
Bounds X: [399000.000, 400000.000]
Bounds Y: [5600000.000, 5601000.000]
Bounds Z: [318.020, 502.340]
Format: las
Point format: 6
LAS version: 1.4
```

### `pywolken pipeline`

Eine JSON-Pipeline-Datei ausführen.

```bash
pywolken pipeline workflow.json
pywolken pipeline workflow.json -v    # Ausführliche Protokollierung
```

### `pywolken convert`

Zwischen Punktwolken-Formaten konvertieren.

```bash
pywolken convert input.laz output.ply
pywolken convert input.laz output.csv
pywolken convert input.ply output.las
```

### `pywolken merge`

Mehrere Dateien zu einer zusammenführen.

```bash
pywolken merge tile1.laz tile2.laz tile3.laz -o merged.laz
```

### `pywolken --version`

Version ausgeben und beenden.

---

## Architektur

### Paketstruktur

```
src/pywolken/
├── __init__.py              # öffentliche API-Exporte
├── _version.py              # Version: "0.1.0"
├── cli.py                   # CLI-Einstiegspunkt
├── parallel.py              # Optionale Dask-Integration
│
├── core/                    # Kern-Datenmodell
│   ├── pointcloud.py        # PointCloud-Klasse (Dict-von-Arrays)
│   ├── dimensions.py        # Standard-Dimensionsdefinitionen
│   ├── bounds.py            # Achsenausgerichtete Bounding-Box
│   └── metadata.py          # Qüll-Metadaten-Container
│
├── io/                      # Reader & Writer
│   ├── base.py              # Abstrakte Reader/Writer-Basisklassen
│   ├── registry.py          # Automatische Formaterkennung, Lese-/Schreibfunktionen
│   ├── las.py               # LAS/LAZ via laspy[lazrs]
│   ├── ply.py               # PLY (ASCII + binär)
│   ├── csv.py               # CSV/TXT/XYZ (Trennzeichen-basierter Text)
│   └── gdal.py              # GeoTIFF-DEM-Writer via rasterio
│
├── filters/                 # 15 Verarbeitungsfilter
│   ├── base.py              # Filter-ABC
│   ├── registry.py          # Filter-Erkennung & -Registrierung
│   ├── range.py             # Dimensionsbereichs-Filterung
│   ├── crop.py              # Räumlicher Bounding-Box-Zuschnitt
│   ├── merge.py             # Mehrere Wolken zusammenführen
│   ├── decimation.py        # Unterabtastung (Schritt/Anteil/Anzahl)
│   ├── assign.py            # Dimensionswerte setzen
│   ├── expression.py        # Boolescher Ausdrucksfilter
│   ├── reprojection.py      # CRS-Transformation
│   ├── ground.py            # SMRF-Bodenklassifikation
│   ├── hag.py               # Höhe über Grund
│   ├── outlier.py           # Statistische/Radius-Ausreisserentfernung
│   ├── normal.py            # PCA-Oberflächennormalen
│   ├── voxel.py             # Voxelraster-Downsampling
│   ├── cluster.py           # DBSCAN-Clustering
│   ├── colorize.py          # RGB aus Raster-Overlay
│   └── sort.py              # Morton/XYZ räumliche Sortierung
│
├── pipeline/                # Pipeline-Engine
│   ├── pipeline.py          # JSON-Pipeline (PDAL-kompatibel)
│   └── streaming.py         # Speicherbegrenzte Chunk-Pipeline
│
├── raster/                  # Rasterverarbeitung
│   ├── dem.py               # DEM-Erzeugung (IDW/Mittelwert/Nächster Nachbar/TIN)
│   ├── hillshade.py         # Schummerung (Horn-Methode)
│   └── export.py            # GeoTIFF-Export via rasterio
│
├── mesh/                    # 3D-Mesh-Erzeugung
│   └── triangulate.py       # 2.5D Delaunay, OBJ/STL/PLY-Export
│
└── utils/                   # Hilfsfunktionen
    └── crs.py               # CRS-Parsing und Reprojektions-Hilfsfunktionen
```

### Designprinzipien

1. **Spaltenorientierte Speicherung:** Jede Dimension ist ein separates NumPy-Array. Hervorragende Cache-Lokalität für Einzeldimensionsoperationen (Filtern, Statistiken). Einfaches Hinzufügen/Entfernen von Dimensionen.

2. **Unveränderliche Filter:** Filter modifizieren niemals die Eingabe-PointCloud. Sie geben immer ein neüs Objekt zurück.

3. **Plugin-Registries:** Reader, Writer und Filter werden über Typnamen registriert. Einfach erweiterbar mit eigenen Implementierungen.

4. **Verzögerte Registrierung:** Integrierte I/O-Handler und Filter werden erst beim ersten Zugriff importiert, um die Importzeit gering zu halten.

5. **PDAL-Kompatibilität:** JSON-Pipeline-Format, Bereichssyntax, Filter-Namenskonventionen und CLI-Muster orientieren sich an PDAL für eine einfache Migration.

6. **Keine C++-Kompilierung:** Alle Abhängigkeiten werden als vorgefertigte Wheels ausgeliefert (lazrs = Rust, NumPy/SciPy = C/Fortran, aber alle über pip verfügbar).

### Eigene Filter hinzufügen

```python
from pywolken.filters.base import Filter
from pywolken.filters.registry import filter_registry
from pywolken.core.pointcloud import PointCloud

class MyFilter(Filter):
    def filter(self, pc: PointCloud) -> PointCloud:
        threshold = float(self.options.get("threshold", 100))
        return pc.mask(pc["Z"] > threshold)

    @classmethod
    def type_name(cls) -> str:
        return "filters.my_custom"

# Registrieren
filter_registry.register(MyFilter)

# Jetzt in Pipelines verwendbar:
# {"type": "filters.my_custom", "threshold": 200}
```

### Eigene I/O-Formate hinzufügen

```python
from pywolken.io.base import Reader, Writer
from pywolken.io.registry import io_registry

class MyReader(Reader):
    def read(self, path, **options):
        # Datei parsen → PointCloud
        ...

    @classmethod
    def extensions(cls):
        return [".myformat"]

    @classmethod
    def type_name(cls):
        return "readers.myformat"

io_registry.register_reader(MyReader)
```

---

## Abhängigkeiten

### Laufzeit-Abhängigkeitsgraph

```
pywolken
├── numpy >= 1.24           (Array-Operationen)
├── laspy[lazrs] >= 2.5     (LAS/LAZ-I/O)
│   └── lazrs               (Rust-LAZ-Codec, vorgefertigtes Wheel)
├── scipy >= 1.10           (KDTree, Delaunay, Interpolation)
└── pyproj >= 3.5           (CRS-Definitionen, Reprojektion)

Optional:
├── rasterio >= 1.3         (GeoTIFF-Export, Raster-Kolorierung)
├── dask[array] >= 2023.0   (Parallelverarbeitung)
├── matplotlib >= 3.7       (2D-Visualisierung)
├── plotly >= 5.0           (3D-Visualisierung)
└── open3d >= 0.17          (Erweiterte Mesh-Verarbeitung)
```

---

## Beispiele

### Vollständiger Workflow: LAZ zu Schummerungs-GeoTIFF

```python
import pywolken
from pywolken.filters.ground import GroundFilter
from pywolken.filters.range import RangeFilter
from pywolken.raster.dem import create_dem
from pywolken.raster.hillshade import hillshade
from pywolken.raster.export import write_geotiff

# 1. LAZ lesen
pc = pywolken.read("terrain.laz")

# 2. Bodenklassifikation
pc = GroundFilter(cell_size=1.0, slope=0.15).filter(pc)

# 3. Bodenpunkte extrahieren
ground = RangeFilter(limits="Classification[2:2]").filter(pc)

# 4. DEM erzeugen
raster, transform = create_dem(ground, resolution=0.5, method="idw")

# 5. Schummerung berechnen
hs = hillshade(raster, resolution=0.5, azimuth=315, altitude=45, z_factor=2.0)

# 6. GeoTIFF exportieren
write_geotiff(hs, "hillshade.tif", transform, nodata=None, dtype="uint8")
```

### JSON-Pipeline: Bodenextraktion

```json
{
  "pipeline": [
    "terrain.laz",
    {"type": "filters.ground", "cell_size": 1.0},
    {"type": "filters.range", "limits": "Classification[2:2]"},
    "ground_only.laz"
  ]
}
```

```bash
pywolken pipeline ground_pipeline.json -v
```

### Formatkonvertierung

```python
import pywolken

# LAS → PLY
pc = pywolken.read("input.laz")
pywolken.write(pc, "output.ply", format="binary_little_endian")

# LAS → CSV
pywolken.write(pc, "output.csv", delimiter=",", precision=3)

# CSV → LAS
pc = pywolken.read("points.csv", header="X,Y,Z,Intensity")
pywolken.write(pc, "output.las")
```

### 3D-Mesh aus Punktwolke

```python
import pywolken
from pywolken.mesh import triangulate_2d

pc = pywolken.read("terrain.laz")
ground = pc.mask(pc["Classification"] == 2)

# Dezimieren für handhabbare Mesh-Grösse
from pywolken.filters.voxel import VoxelFilter
ground_voxel = VoxelFilter(cell_size=2.0).filter(ground)

# Triangulieren und exportieren
mesh = triangulate_2d(ground_voxel, max_edge_length=10.0)
mesh.write("terrain.obj")
mesh.write("terrain.stl")
mesh.write("terrain.ply")

# Mesh weiter dezimieren
small = mesh.decimate(target_faces=50000)
small.write("terrain_simple.obj")
```

### Streaming grosser Dateien

```python
from pywolken import StreamingPipeline
from pywolken.io.las import LasReader, LasWriter
from pywolken.filters.range import RangeFilter
from pywolken.filters.decimation import DecimationFilter

sp = StreamingPipeline(
    reader=LasReader(),
    input_path="huge_4gb.laz",
    writer=LasWriter(),
    output_path="filtered.laz",
    filters=[
        RangeFilter(limits="Classification[2:2]"),
        DecimationFilter(step=5),
    ],
    chunk_size=2_000_000,
)
total = sp.execute()
print(f"Gelesen: {sp.total_read:,} → Geschrieben: {sp.total_written:,} Punkte")
```

### Parallelverarbeitung mit Dask

```python
from pywolken.parallel import parallel_read, parallel_filter
from pywolken.filters.range import RangeFilter

# Mehrere Kacheln parallel lesen
pc = parallel_read([
    "tile_001.laz", "tile_002.laz", "tile_003.laz",
    "tile_004.laz", "tile_005.laz", "tile_006.laz",
])

# In parallelen Chunks filtern
result = parallel_filter(
    pc,
    filters=[RangeFilter(limits="Classification[2:6]")],
    n_chunks=8,
)
```

### Verkettete Filterverarbeitung

```python
import pywolken
from pywolken.filters.ground import GroundFilter
from pywolken.filters.hag import HagFilter
from pywolken.filters.outlier import OutlierFilter
from pywolken.filters.expression import ExpressionFilter

pc = pywolken.read("terrain.laz")

# Kette: Ausreisserentfernung → Boden → HAG → hohe Objekte filtern
pc = OutlierFilter(method="statistical", mean_k=8, multiplier=2.0).filter(pc)
pc = GroundFilter().filter(pc)
pc = HagFilter().filter(pc)
tall = ExpressionFilter(expression="HeightAboveGround > 20").filter(pc)

print(f"{tall.num_points:,} Punkte über 20m gefunden")
pywolken.write(tall, "tall_objects.laz")
```

### Punktwolken-Statistiken

```python
import pywolken
import numpy as np

pc = pywolken.read("terrain.laz")

print(f"Punkte:      {pc.num_points:,}")
print(f"CRS:         {pc.crs}")
print(f"Bounds:      {pc.bounds}")
print(f"Z-Bereich:   {pc['Z'].min():.2f} - {pc['Z'].max():.2f} m")
print(f"Mittel Z:    {pc['Z'].mean():.2f} m")
print(f"Dimensionen: {pc.dimensions}")

# Klassifikationsaufschlüsselung
if "Classification" in pc:
    from pywolken.core.dimensions import CLASSIFICATION_CODES
    classes, counts = np.uniqü(pc["Classification"], return_counts=Trü)
    for cls, count in zip(classes, counts):
        name = CLASSIFICATION_CODES.get(int(cls), "Unknown")
        print(f"  Klasse {cls} ({name}): {count:,}")
```
