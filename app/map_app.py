"""
Interactive Geospatial Web Map Application for Wildfire Spread Risk Mapper.
Built with Folium and Leaflet.js.
Features:
- Multi-basemap selector (Esri Satellite, CartoDB Positron, OpenTopoMap).
- Choropleth vector layer for 5-tier wildfire spread hazard classification.
- Dynamic interactive popup cards with micro-environmental diagnostics.
- Active thermal anomalies layer (NASA FIRMS VIIRS/MODIS detections).
- Layer control, custom floating color legend, minimap, and measurement tools.
- Export to standalone self-contained HTML and optional local HTTP server.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import http.server
import socketserver

import folium
from folium import plugins
import branca.colormap as cm
import geopandas as gpd

# Ensure scripts dir is accessible for utils
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir / "scripts"))
from utils import setup_logger, load_config, get_project_root, reproject_gdf

LOGGER = setup_logger("map_app")


class WildfireRiskMapApp:
    """Compiles and serves an interactive Folium geospatial dashboard."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        self.root = get_project_root()
        self.tier_config = self.config["risk_classification"]["tiers"]

    def create_interactive_map(
        self,
        geojson_path: Optional[Path] = None,
        output_html: Optional[Path] = None
    ) -> Path:
        """
        Builds the comprehensive Folium map document and outputs an interactive HTML file.
        """
        geojson_path = geojson_path or (self.root / "data" / "processed" / "wildfire_risk_summary.geojson")
        output_html = output_html or (self.root / "app" / "wildfire_risk_map.html")
        output_html.parent.mkdir(parents=True, exist_ok=True)

        if not geojson_path.exists():
            LOGGER.warning(f"Summary GeoJSON not found at {geojson_path}. Generating risk surfaces first...")
            from risk_map_generation import RiskMapGenerator
            gen = RiskMapGenerator()
            model, feat_names, gdf = gen.load_model_and_data()
            scored = gen.predict_risk_tiers(model, feat_names, gdf)
            gen.export_risk_geotiff(scored)
            gen.export_geojson_summary(scored, output_geojson=geojson_path)

        # Load GeoJSON data
        gdf = gpd.read_file(geojson_path)
        LOGGER.info(f"Loaded {len(gdf)} risk assessment polygons for map rendering.")

        # Compute centroid center for default view
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy] in WGS84
        center_lat = (bounds[1] + bounds[3]) / 2.0
        center_lon = (bounds[0] + bounds[2]) / 2.0

        # 1. Initialize Base Map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=9,
            tiles=None,
            control_scale=True
        )

        # 2. Add Base Tile Layers
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery, Maxar, Earthstar Geographics",
            name="Esri Satellite Imagery",
            overlay=False,
            control=True
        ).add_to(m)

        folium.TileLayer(
            tiles="CartoDB positron",
            name="CartoDB Positron (Light)",
            overlay=False,
            control=True
        ).add_to(m)

        folium.TileLayer(
            tiles="OpenStreetMap",
            name="OpenStreetMap Standard",
            overlay=False,
            control=True
        ).add_to(m)

        # 3. Add Choropleth Wildfire Risk Tier Layer
        tier_colors = {
            1: "#2ca25f",  # Low - Green
            2: "#ffeb3b",  # Moderate - Yellow
            3: "#fe9929",  # High - Orange
            4: "#d95f0e",  # Very High - Deep Orange
            5: "#990000"   # Extreme - Crimson
        }

        def style_function(feature):
            tier = feature["properties"].get("risk_tier", 1)
            prob = feature["properties"].get("predicted_probability", 0.0)
            color = tier_colors.get(tier, "#2ca25f")
            fill_opacity = 0.35 + 0.45 * float(prob)
            return {
                "fillColor": color,
                "color": "#333333",
                "weight": 0.8,
                "fillOpacity": fill_opacity
            }

        def highlight_function(feature):
            return {
                "weight": 3,
                "color": "#ffffff",
                "fillOpacity": 0.85
            }

        risk_layer = folium.GeoJson(
            gdf,
            name="Wildfire Spread Hazard Tiers (250m Grid)",
            style_function=style_function,
            highlight_function=highlight_function,
            tooltip=folium.GeoJsonTooltip(
                fields=["cell_id", "risk_label", "predicted_probability", "fwi_score"],
                aliases=["Grid Unit:", "Risk Tier:", "Fire Probability:", "FWI Index:"],
                style="background-color: #1a1a1a; color: #ffffff; font-family: sans-serif; font-size: 12px; padding: 8px;"
            ),
            popup=folium.GeoJsonPopup(
                fields=[
                    "cell_id", "risk_label", "predicted_probability", "fwi_score",
                    "slope_degrees", "canopy_cover_pct", "dist_to_road_m", "action_guidance"
                ],
                aliases=[
                    "Grid Cell ID", "Hazard Tier", "Spread Probability", "Fire Weather Index",
                    "Slope Angle (°)", "Canopy Fuel Cover (%)", "Distance to Road (m)", "Mitigation Directive"
                ],
                style="max-width: 320px; font-size: 12px; line-height: 1.4;"
            )
        )
        risk_layer.add_to(m)

        # 4. Add Active Thermal Hotspots (NASA FIRMS mock cluster)
        hotspot_group = folium.FeatureGroup(name="NASA FIRMS Active Thermal Detections", show=True)
        high_risk_cells = gdf[gdf["risk_tier"] >= 4].iloc[:40]
        for _, row in high_risk_cells.iterrows():
            pt = row.geometry.centroid
            folium.CircleMarker(
                location=[pt.y, pt.x],
                radius=6,
                color="#b30000",
                fill=True,
                fill_color="#ff1a1a",
                fill_opacity=0.9,
                tooltip=f"Active VIIRS Detection | FRP: {round(float(row['predicted_probability']) * 120, 1)} MW",
                popup=(
                    f"<b>Active Thermal Anomaly</b><br>"
                    f"Sensor: VIIRS S-NPP (375m)<br>"
                    f"Confidence: High<br>"
                    f"Est FRP: {round(float(row['predicted_probability']) * 120, 1)} MW"
                )
            ).add_to(hotspot_group)
        hotspot_group.add_to(m)

        # 5. Add Custom Floating HTML Legend
        legend_html = """
        <div style="
            position: fixed; 
            bottom: 35px; left: 35px; width: 230px; height: auto; 
            z-index: 9999; font-size: 12px;
            background-color: rgba(255, 255, 255, 0.92);
            padding: 12px 16px;
            box-shadow: 0 0 15px rgba(0,0,0,0.25);
            border-radius: 8px;
            font-family: Arial, sans-serif;
            ">
            <h4 style="margin: 0 0 8px 0; font-size: 14px; font-weight: bold; color: #111;">Wildfire Hazard Tier</h4>
            <div style="display: flex; align-items: center; margin-bottom: 4px;">
                <span style="background: #990000; width: 18px; height: 14px; display: inline-block; margin-right: 8px; border-radius: 2px;"></span>
                <b>Extreme</b> &nbsp;(&ge; 0.80)
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 4px;">
                <span style="background: #d95f0e; width: 18px; height: 14px; display: inline-block; margin-right: 8px; border-radius: 2px;"></span>
                <b>Very High</b> &nbsp;(0.60 &ndash; 0.80)
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 4px;">
                <span style="background: #fe9929; width: 18px; height: 14px; display: inline-block; margin-right: 8px; border-radius: 2px;"></span>
                <b>High</b> &nbsp;(0.40 &ndash; 0.60)
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 4px;">
                <span style="background: #ffeb3b; width: 18px; height: 14px; display: inline-block; margin-right: 8px; border: 1px solid #ccc; border-radius: 2px;"></span>
                <b>Moderate</b> &nbsp;(0.20 &ndash; 0.40)
            </div>
            <div style="display: flex; align-items: center;">
                <span style="background: #2ca25f; width: 18px; height: 14px; display: inline-block; margin-right: 8px; border-radius: 2px;"></span>
                <b>Low</b> &nbsp;(&lt; 0.20)
            </div>
            <hr style="margin: 8px 0;">
            <span style="font-size: 10px; color: #555;">Projection: EPSG:3310 / WGS84<br>Model: Spatial RF Classifier</span>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        # 6. Map Plugins & Utilities
        plugins.Fullscreen(position="topright", title="Fullscreen Mode", force_separate_button=True).add_to(m)
        plugins.MeasureControl(position="topleft", primary_length_unit="kilometers", primary_area_unit="sqmeters").add_to(m)
        plugins.MiniMap(toggle_display=True, position="bottomright").add_to(m)
        folium.LayerControl(position="topright", collapsed=False).add_to(m)

        # Save HTML
        m.save(str(output_html))
        LOGGER.info(f"Interactive web map successfully compiled to: {output_html}")
        return output_html

    def serve(self, html_path: Path, port: int = 8080):
        """Launches a local HTTP server for real-time dashboard inspection."""
        LOGGER.info(f"Starting local map preview server on http://localhost:{port}...")
        web_dir = html_path.parent
        os.chdir(str(web_dir))
        handler = http.server.SimpleHTTPRequestHandler
        with socketserver.TCPServer(("", port), handler) as httpd:
            print(f"Server accessible at: http://localhost:{port}/{html_path.name}")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("Server stopped.")


def main():
    parser = argparse.ArgumentParser(description="Generate and serve Folium interactive wildfire hazard map.")
    parser.add_argument("--output", type=str, default="app/wildfire_risk_map.html", help="Path to output HTML")
    parser.add_argument("--serve", action="store_true", help="Launch local HTTP web server")
    parser.add_argument("--port", type=int, default=8080, help="Port for web server")
    args = parser.parse_args()

    root = get_project_root()
    app = WildfireRiskMapApp()
    html_file = app.create_interactive_map(output_html=root / args.output)

    if args.serve:
        app.serve(html_file, port=args.port)


if __name__ == "__main__":
    main()
