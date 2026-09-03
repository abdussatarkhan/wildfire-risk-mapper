-- =============================================================================
-- Wildfire Spread Risk Mapper - Spatial SQL & PostGIS Analysis Queries
-- Target System: PostgreSQL 14+ with PostGIS 3.2+
-- Coordinate Reference System: EPSG:3310 (California Albers Equal Area)
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. Table Definitions & Spatial Index Configuration
-- -----------------------------------------------------------------------------

-- Regular 250m Analysis Fishnet Grid
CREATE TABLE IF NOT EXISTS wildfire_grid_250m (
    cell_id VARCHAR(32) PRIMARY KEY,
    block_id VARCHAR(32) NOT NULL,
    grid_col INTEGER NOT NULL,
    grid_row INTEGER NOT NULL,
    centroid_x DOUBLE PRECISION NOT NULL,
    centroid_y DOUBLE PRECISION NOT NULL,
    geom GEOMETRY(Polygon, 3310) NOT NULL,
    centroid_geom GEOMETRY(Point, 3310) GENERATED ALWAYS AS (ST_Centroid(geom)) STORED
);
CREATE INDEX IF NOT EXISTS idx_wildfire_grid_geom ON wildfire_grid_250m USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_wildfire_grid_centroid ON wildfire_grid_250m USING GIST (centroid_geom);
CREATE INDEX IF NOT EXISTS idx_wildfire_grid_block ON wildfire_grid_250m (block_id);

-- NASA FIRMS Active Thermal Detections
CREATE TABLE IF NOT EXISTS firms_active_fires (
    detection_id BIGSERIAL PRIMARY KEY,
    sensor VARCHAR(16) NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    bright_ti4 DOUBLE PRECISION,
    frp_mw DOUBLE PRECISION NOT NULL,
    confidence VARCHAR(16) NOT NULL,
    acq_date DATE NOT NULL,
    acq_time VARCHAR(8) NOT NULL,
    daynight CHAR(1),
    geom GEOMETRY(Point, 3310) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_firms_geom ON firms_active_fires USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_firms_date ON firms_active_fires (acq_date);
CREATE INDEX IF NOT EXISTS idx_firms_frp ON firms_active_fires (frp_mw);

-- Primary and Secondary Transportation Network (TIGER/Line)
CREATE TABLE IF NOT EXISTS tiger_road_network (
    road_id VARCHAR(32) PRIMARY KEY,
    road_name VARCHAR(128),
    road_class VARCHAR(32), -- Primary, Secondary, Interstate
    speed_limit_mph INTEGER,
    geom GEOMETRY(MultiLineString, 3310) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tiger_roads_geom ON tiger_road_network USING GIST (geom);

-- Model-Scored Wildfire Risk Output Surfaces
CREATE TABLE IF NOT EXISTS wildfire_risk_surfaces (
    cell_id VARCHAR(32) PRIMARY KEY REFERENCES wildfire_grid_250m(cell_id),
    predicted_probability NUMERIC(5, 4) NOT NULL,
    risk_tier SMALLINT NOT NULL CHECK (risk_tier BETWEEN 1 AND 5),
    risk_label VARCHAR(16) NOT NULL,
    fwi_score NUMERIC(5, 2),
    slope_degrees NUMERIC(4, 1),
    canopy_cover_pct NUMERIC(4, 1),
    dist_to_road_m NUMERIC(8, 2),
    pop_density_km2 NUMERIC(8, 2),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_risk_tier ON wildfire_risk_surfaces (risk_tier);
CREATE INDEX IF NOT EXISTS idx_risk_prob ON wildfire_risk_surfaces (predicted_probability DESC);


-- -----------------------------------------------------------------------------
-- 2. Spatial Join & Proximity Analysis Queries
-- -----------------------------------------------------------------------------

-- Query 2.1: Point-in-Polygon Historical FIRMS Fire Detection Aggregation
-- Aggregates 10-year thermal detection counts and total radiative energy per 250m cell
SELECT 
    g.cell_id,
    g.block_id,
    COUNT(f.detection_id) AS historical_fire_count,
    ROUND(COALESCE(MAX(f.frp_mw), 0)::numeric, 2) AS max_frp_mw,
    ROUND(COALESCE(AVG(f.frp_mw), 0)::numeric, 2) AS avg_frp_mw,
    ROUND(COALESCE(SUM(f.frp_mw), 0)::numeric, 2) AS cumulative_radiative_power_mw
FROM wildfire_grid_250m g
LEFT JOIN firms_active_fires f 
    ON ST_Contains(g.geom, f.geom)
    AND f.acq_date >= (CURRENT_DATE - INTERVAL '10 years')
GROUP BY g.cell_id, g.block_id;

-- Query 2.2: Vector Proximity - Exact Euclidean Distance to Nearest Primary Highway
-- Utilizes PostGIS KNN (<->) bounding box operator with lateral join for rapid distance computation
SELECT 
    g.cell_id,
    ROUND(ST_Distance(g.centroid_geom, r.geom)::numeric, 2) AS dist_to_nearest_highway_m,
    r.road_id,
    r.road_name,
    r.road_class
FROM wildfire_grid_250m g
CROSS JOIN LATERAL (
    SELECT road_id, road_name, road_class, geom
    FROM tiger_road_network
    ORDER BY g.centroid_geom <-> geom
    LIMIT 1
) r;

-- Query 2.3: 5km Spatial Buffer Kernel Anomaly Density
-- Calculates total fire detection density within a 5,000-meter radius around each cell centroid
SELECT 
    g.cell_id,
    COUNT(f.detection_id) AS detections_within_5km,
    ROUND(SUM(f.frp_mw)::numeric, 2) AS total_frp_within_5km
FROM wildfire_grid_250m g
JOIN firms_active_fires f 
    ON ST_DWithin(g.centroid_geom, f.geom, 5000)
GROUP BY g.cell_id;


-- -----------------------------------------------------------------------------
-- 3. Wildland-Urban Interface (WUI) & Community Vulnerability Analysis
-- -----------------------------------------------------------------------------

-- Query 3.1: High and Extreme Hazard Exposure in Wildland-Urban Interface
-- Identifies populated cells within High or Extreme risk zones requiring evacuation pre-planning
CREATE OR REPLACE VIEW view_wui_critical_hazard_cells AS
SELECT 
    g.cell_id,
    g.block_id,
    r.risk_tier,
    r.risk_label,
    r.predicted_probability,
    r.pop_density_km2,
    r.dist_to_road_m,
    r.slope_degrees,
    r.canopy_cover_pct,
    -- Est. population per 250m x 250m cell (0.0625 km^2)
    ROUND((r.pop_density_km2 * 0.0625)::numeric, 1) AS estimated_residents,
    g.geom
FROM wildfire_grid_250m g
JOIN wildfire_risk_surfaces r ON g.cell_id = r.cell_id
WHERE r.risk_tier IN (4, 5) -- Tier 4: Very High, Tier 5: Extreme
  AND r.pop_density_km2 > 5.0;

-- Query 3.2: Critical Evacuation Routes at Risk of Encroachment
-- Finds highway segments intersecting or within 150m of Extreme wildfire hazard zones
SELECT 
    rd.road_id,
    rd.road_name,
    rd.road_class,
    COUNT(g.cell_id) AS extreme_hazard_adjacent_cells,
    ROUND(ST_Length(rd.geom)::numeric, 2) AS total_corridor_length_m,
    ST_Union(g.geom) AS hazard_cluster_footprint
FROM tiger_road_network rd
JOIN wildfire_grid_250m g 
    ON ST_DWithin(rd.geom, g.geom, 150)
JOIN wildfire_risk_surfaces r 
    ON g.cell_id = r.cell_id
WHERE r.risk_tier = 5 -- Extreme Risk
GROUP BY rd.road_id, rd.road_name, rd.road_class, rd.geom
HAVING COUNT(g.cell_id) >= 3
ORDER BY extreme_hazard_adjacent_cells DESC;


-- -----------------------------------------------------------------------------
-- 4. Spatial Autocorrelation & Regional Aggregations
-- -----------------------------------------------------------------------------

-- Query 4.1: Spatial Lag of Wildfire Spread Probability (Local Spatial Moving Average)
-- Calculates the average predicted probability of the 8 nearest neighboring grid cells
SELECT 
    g1.cell_id,
    r1.predicted_probability AS local_prob,
    ROUND(AVG(r2.predicted_probability)::numeric, 4) AS spatial_lag_prob
FROM wildfire_grid_250m g1
JOIN wildfire_risk_surfaces r1 ON g1.cell_id = r1.cell_id
CROSS JOIN LATERAL (
    SELECT r2.predicted_probability
    FROM wildfire_grid_250m g2
    JOIN wildfire_risk_surfaces r2 ON g2.cell_id = r2.cell_id
    WHERE g1.cell_id <> g2.cell_id
    ORDER BY g1.centroid_geom <-> g2.centroid_geom
    LIMIT 8
) r2
GROUP BY g1.cell_id, r1.predicted_probability;

-- Query 4.2: Regional Planning Block Hazard Summary
-- Computes acreage distribution and population at risk by 25km planning block
SELECT 
    g.block_id,
    COUNT(g.cell_id) AS total_cells,
    -- 250m x 250m = 62,500 m^2 = 15.444 acres per cell
    ROUND((COUNT(g.cell_id) * 15.444)::numeric, 1) AS total_acreage,
    COUNT(CASE WHEN r.risk_tier = 5 THEN 1 END) AS extreme_risk_cells,
    COUNT(CASE WHEN r.risk_tier = 4 THEN 1 END) AS very_high_risk_cells,
    COUNT(CASE WHEN r.risk_tier = 3 THEN 1 END) AS high_risk_cells,
    ROUND((COUNT(CASE WHEN r.risk_tier >= 4 THEN 1 END)::numeric / COUNT(g.cell_id)::numeric) * 100, 1) AS pct_critical_hazard,
    ROUND(SUM(r.pop_density_km2 * 0.0625)::numeric, 0) AS total_residents_exposed
FROM wildfire_grid_250m g
JOIN wildfire_risk_surfaces r ON g.cell_id = r.cell_id
GROUP BY g.block_id
ORDER BY pct_critical_hazard DESC;
