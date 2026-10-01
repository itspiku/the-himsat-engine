# Data sources and licences

| Data | Source | Licence / terms | Used for |
|---|---|---|---|
| Sentinel-2 L2A | Copernicus, via Microsoft Planetary Computer (`sentinel-2-l2a`) or Element84 Earth Search | Copernicus open data licence (free, full and open; attribution "Contains modified Copernicus Sentinel data [year]") | lakes, snow/ice, optical change, cracks, turbidity |
| Sentinel-1 RTC | Copernicus, radiometrically terrain-corrected by Planetary Computer (`sentinel-1-rtc`) | Copernicus open data licence; RTC processing © Microsoft/Catalyst, CC BY 4.0 | velocity, mass movements, lake area under cloud |
| Copernicus DEM GLO-30 | ESA / Copernicus via Planetary Computer (`cop-dem-glo-30`) | © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018, provided under COPERNICUS by the EU and ESA; free licence | slope, shadows, flow routing, exposure |
| OpenStreetMap (assets, glacier outlines) | Overpass API, cached in `data/osm/` | © OpenStreetMap contributors, ODbL 1.0: attribute, share-alike for derived databases | settlements, hydropower, bridges, schools, glacier inventory |
| Curated infrastructure (`data/assets/curated.geojson`) | compiled from public reporting; approximate | CC0 | border crossing and hydropower headworks missing from OSM |
| Prithvi-EO-2.0-100M-TL | NASA / IBM on Hugging Face | Apache-2.0 | segmentation backbone |
| Segment Anything (SAM) | Meta AI (`facebook/sam-vit-base`) | Apache-2.0 | lake outline refinement |
| Qwen2.5 / Llama 3.x (optional) | Alibaba / Meta | Qwen licence / Llama community licence | alert wording |
| Basemaps in the web map | OpenStreetMap tiles; EOX Sentinel-2 cloudless 2016 | OSM tile usage policy; CC BY 4.0 (2016 mosaic) | display only |

**Production note:** the public OSM tile servers are not meant for heavy production traffic.
Point the map at your own tile service or a commercial provider (see `web/src/map.ts`) before a
public launch.
