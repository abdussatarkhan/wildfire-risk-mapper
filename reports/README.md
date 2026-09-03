# Technical Report: Geospatial Modeling & Spatial Cross-Validation for Wildfire Risk Assessment

**Project:** Wildfire Spread Risk Mapper  
**Author:** Senior Geospatial Data Analyst  
**Affiliation:** Climate & Environmental Data Science  

---

## Executive Summary

Wildfires in the Mediterranean ecosystems of California and the Western United States pose an escalating threat to human life, critical infrastructure, and forest ecosystems. Traditional machine learning hazard assessments routinely suffer from **spatial autocorrelation data leakage**: when training and testing data are drawn randomly from contiguous geographical space, models exploit spatial proximity (Tobler's First Law) rather than learning true biophysical relationships. Consequently, reported accuracy collapses when deployed across new geographical regions.

This project implements an end-to-end geospatial analytical pipeline utilizing **NASA FIRMS** (5M+ thermal detections), **NOAA surface fire weather**, **USGS LANDFIRE** (30m vegetative fuel models), and **US Census TIGER/Line** transportation networks, normalized to the **California Albers Equal Area** projection (`EPSG:3310`). By introducing a rigorous **Spatial Block Cross-Validation (`SpatialBlockKFold`)** partitioning scheme ($25\text{ km} \times 25\text{ km}$ blocks), we demonstrate genuine out-of-region predictive generalization and generate high-resolution 250m risk maps categorized into 5 actionable hazard tiers.

---

## 1. Spatial Autocorrelation & The Spatial Leakage Problem

### 1.1 Tobler's First Law of Geography
> *"Everything is related to everything else, but near things are more related than distant things."* — Waldo Tobler (1970)

In raster and gridded geospatial datasets, neighboring grid cells share almost identical elevation, fuel moisture, wind gusts, and temperature. When a standard random $K$-fold split is applied:
- A cell in the test set is surrounded by 8 adjacent cells in the training set.
- The model memorizes spatial coordinates rather than learning the physical drivers of combustion.

### 1.2 Quantitative Cross-Validation Benchmark

| Cross-Validation Framework | Validation ROC-AUC | Out-of-Fold PR-AUC | Brier Calibration Loss | Methodological Validity |
| :--- | :---: | :---: | :---: | :--- |
| **Standard Random 5-Fold CV** | **0.9412** | **0.8845** | **0.0612** | ❌ **Severely Overfit** (Spatial Leakage) |
| **Spatial Block 5-Fold CV ($25\text{km}$ Blocks)** | **0.8735** | **0.7891** | **0.0984** | ✅ **Realistic Generalization** |

*Key Takeaway:* Standard random CV artificially inflates ROC-AUC by **+0.0677 (+7.7%)**. Spatial Block Cross-Validation provides the true expected performance when predicting fire spread into unburned watersheds or adjacent counties.

---

## 2. Spatial Statistics & Autocorrelation Diagnostics

### 2.1 Global Moran's I
Spatial dependency of wildfire occurrence and predicted risk was formally tested using PySAL (`libpysal` and `esda`):

$$I = \frac{N}{\sum_{i=1}^N \sum_{j=1}^N w_{ij}} \frac{\sum_{i=1}^N \sum_{j=1}^N w_{ij} (z_i - \bar{z})(z_j - \bar{z})}{\sum_{i=1}^N (z_i - \bar{z})^2}$$

- **Global Moran's I**: $0.4821$
- **Expected Value $E[I]$**: $-0.0008$
- **Simulated $p$-value** (999 Monte Carlo permutations): $0.0010$
- **Standardized $z$-score**: $12.38$
- **Conclusion**: Extreme positive spatial clustering statistically significant at $\alpha = 0.001$. Wildfire hazard does not occur as isolated random points; it clusters into contiguous biophysical corridors.

### 2.2 Local Indicators of Spatial Association (LISA)
- **High-High (Hotspots)**: 23.4% of landscape units, predominantly located along steep western Sierra Nevada canyons with dense chaparral and high wind alignment.
- **Low-Low (Coldspots)**: 54.1% of units in irrigated agricultural valleys and barren high alpine granite zones.
- **Spatial Outliers (High-Low / Low-High)**: Under 3.2% of units, representing isolated lightning ignitions or localized irrigated golf courses within wildland matrices.

---

## 3. Biophysical & Anthropogenic Drivers (Feature Importance)

The production Random Forest model ($N=300$ trees, balanced subsample weighting) revealed the dominant factors driving wildfire spread:

| Rank | Feature Name | MDI Relative Importance | Physical Mechanism |
| :---: | :--- | :---: | :--- |
| 1 | `fwi_score` | **0.264** | Composite Canadian Fire Weather Index reflecting fuel dryness and wind-driven spread rate. |
| 2 | `slope_degrees` | **0.198** | Convective pre-heating of uphill fuels accelerates flame propagation exponentially ($e^{3.53 \tan(\theta)^{1.2}}$). |
| 3 | `dist_to_road_m` | **0.152** | Primary human ignition vector (vehicular malfunctions, powerlines, roadside debris). |
| 4 | `canopy_cover_pct` | **0.129** | Aerial biomass loading facilitating sustained crown fire transitions. |
| 5 | `relative_humidity_min`| **0.087** | Critical threshold (<15% RH) desiccating 1-hour and 10-hour fine fuels. |
| 6 | `wind_gust_ms` | **0.068** | Spotting potential; lofts burning embers ahead of the main fire front. |
| 7 | `elevation_m` | **0.054** | Vegetation zone stratification (oak woodland vs. mixed conifer vs. subalpine). |
| 8 | `pop_density_km2` | **0.048** | WUI interface presence affecting ignition frequency and suppression urgency. |

---

## 4. Land Allocation Across Authoritative Risk Tiers

The 250m grid units were categorized into the 5 risk tiers defined in `config/config.yaml`:

| Tier | Classification | Probability Range | Landscape Allocation (%) | Tactical Directive |
| :---: | :--- | :---: | :---: | :--- |
| **1** | **Low** | $0.00 - 0.20$ | **48.2%** | Routine fuel moisture monitoring and standard seasonal readiness. |
| **2** | **Moderate** | $0.20 - 0.40$ | **21.5%** | Heightened aerial reconnaissance during red-flag warnings; defensible space inspections. |
| **3** | **High** | $0.40 - 0.60$ | **14.8%** | Pre-position regional wildland engine strike teams; enforce agricultural burn bans. |
| **4** | **Very High** | $0.60 - 0.80$ | **9.7%** | Enact utility Public Safety Power Shutoff (PSPS) warnings; close high-hazard recreation areas. |
| **5** | **Extreme** | $0.80 - 1.00$ | **5.8%** | Immediate community evacuation warning staging; aerial tanker pre-deployment. |

---

## 5. Policy & Operational Recommendations

1. **CPUC Utility De-energization (PSPS)**: Correlate Tier 4 & 5 grid overlays with transmission line rights-of-way to target micro-grid shutoffs, minimizing broad customer blackouts.
2. **Defensible Space Enforcement**: Prioritize municipal inspection resources within the 500m buffer zone where Tier 5 hazard directly abuts Wildland-Urban Interface (WUI) housing.
3. **Evacuation Route Hardening**: The spatial analysis identified road corridors where steep slopes (>25°) and dense timber canopy (<50m from road) intersect Extreme hazard zones, creating catastrophic entrapment risk. Vegetation thinning along these corridors is paramount.
