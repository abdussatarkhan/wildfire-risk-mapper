# Geospatial Wildfire Spread Risk & Hazard Mapping System

[![GeoPandas](https://img.shields.io/badge/GIS-GeoPandas-2E7D32?style=for-the-badge&logo=python&logoColor=white)](https://geopandas.org/) [![Rasterio](https://img.shields.io/badge/Raster-Rasterio-00A8E8?style=for-the-badge)](https://rasterio.readthedocs.io/) [![Python](https://img.shields.io/badge/Python-Geospatial_AI-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Author](https://img.shields.io/badge/Author-Abdussatar-E50914?style=for-the-badge&logo=github&logoColor=white)](https://github.com/abdussatarkhan)

> **A geospatial predictive hazard model integrating satellite multispectral imagery (NDVI/EVI), meteorological indices (VPD, temperature, wind), and spatial autocorrelation (Moran's I) to forecast high-resolution wildfire ignition risks.**

---

## 🏛️ System Architecture

```mermaid
graph TD
    Satellite[Satellite Imagery: NDVI / Soil Moisture] --> GIS[GeoPandas & Rasterio Grid Processing]
    Weather[ERA5 Climate & Wind Vectors] --> GIS
    GIS --> SpatialML[Spatial Autocorrelation Moran's I & Random Forest]
    SpatialML --> HazardMap[Interactive Geospatial Risk Heatmap]
```

---

## 🌟 Key Features & Capabilities

- **Production-Grade Implementation**: Built with high attention to performance, modular design, and industry standard best practices.
- **Enterprise Data Architecture**: Scalable data schemas, reproducible synthetic generators, and optimized queries.
- **Explainable & Validated**: Comprehensive evaluation metrics, error analyses, and validation tests.
- **Comprehensive Tech Stack**: `Python` `GeoPandas` `Rasterio` `Shapely` `Scikit-Learn` `GIS`.

---

## 📊 Visual Preview & Analysis

<div align="center">

![wildfire-risk-mapper preview](images/wildfire_hazard_map.png)

</div>

---

## 🚀 Quickstart & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/abdussatarkhan/wildfire-risk-mapper.git
cd wildfire-risk-mapper
```

### 2. Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies (if requirements.txt exists)
pip install -r requirements.txt
```

---

## 👨‍💻 Author & Profile

Built and maintained by **Abdussatar** ([@abdussatarkhan](https://github.com/abdussatarkhan)).  
For technical discussions, collaboration, or queries, feel free to reach out via [LinkedIn](https://www.linkedin.com/in/abdus-satar-5150813b5/) or [GitHub](https://github.com/abdussatarkhan).

---

## 📜 License

This project is licensed under the **MIT License** — see the LICENSE file for details.
