import maplibregl, { type GeoJSONSource, type Map as MLMap, type StyleSpecification } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";

export const LEVEL_COLOR: Record<string, string> = {
  high: "#d03b3b", // status: critical
  medium: "#ec835a", // status: serious
  low: "#0ca30c", // status: good
  unknown: "#898781",
};

const LEVEL_EXPR = ["match", ["get", "level"], "high", LEVEL_COLOR.high, "medium", LEVEL_COLOR.medium, "low",
  LEVEL_COLOR.low, LEVEL_COLOR.unknown] as unknown as maplibregl.ExpressionSpecification;

const STYLE: StyleSpecification = {
  version: 8,
  glyphs: "https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf",
  sources: {
    osm: {
      type: "raster",
      tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
      tileSize: 256,
      maxzoom: 19,
      attribution: "© OpenStreetMap contributors",
    },
    s2: {
      type: "raster",
      tiles: ["https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless_3857/default/g/{z}/{y}/{x}.jpg"],
      tileSize: 256,
      maxzoom: 15,
      attribution: 'Sentinel-2 cloudless – <a href="https://s2maps.eu">s2maps.eu</a> by EOX (Copernicus data 2016, CC BY 4.0)',
    },
  },
  layers: [
    { id: "bg", type: "background", paint: { "background-color": "#e9e7e1" } },
    { id: "osm", type: "raster", source: "osm" },
    { id: "s2", type: "raster", source: "s2", layout: { visibility: "none" } },
  ],
};

const EMPTY: GeoJSON.FeatureCollection = { type: "FeatureCollection", features: [] };

const ready = new WeakMap<MLMap, boolean>();

/** Run ``fn`` now if the map's sources exist, otherwise as soon as they do. */
export function onReady(map: MLMap, fn: () => void): void {
  if (ready.get(map)) fn();
  else map.once("himsat:ready" as "load", fn);
}

/** Create the map immediately; data can be pushed before it has finished loading. */
export function createMap(container: HTMLElement): MLMap {
  const map = new maplibregl.Map({
    container,
    style: STYLE,
    center: [85.45, 28.25],
    zoom: 9.3,
    attributionControl: { compact: true },
    cooperativeGestures: false,
  });
  map.addControl(new maplibregl.NavigationControl({ visualizePitch: false }), "top-left");
  // keep the canvas in step with its container (layout changes, panel resizes, orientation)
  new ResizeObserver(() => map.resize()).observe(container);
  map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-left");
  {
    map.on("load", () => {
      map.addSource("sites", { type: "geojson", data: EMPTY });
      map.addSource("selected-path", { type: "geojson", data: EMPTY });
      map.addSource("exposed", { type: "geojson", data: EMPTY });
      map.addSource("events", { type: "geojson", data: EMPTY });
      map.addSource("event-point", { type: "geojson", data: EMPTY });

      map.addLayer({ id: "events-fill", type: "fill", source: "events", filter: ["==", ["geometry-type"], "Polygon"],
        layout: { visibility: "none" }, paint: { "fill-color": "#7a4fd0", "fill-opacity": 0.35 } });
      map.addLayer({ id: "events-dot", type: "circle", source: "events", filter: ["==", ["geometry-type"], "Point"],
        layout: { visibility: "none" }, paint: { "circle-color": "#7a4fd0", "circle-radius": 5, "circle-opacity": 0.8,
          "circle-stroke-color": "#ffffff", "circle-stroke-width": 1 } });
      map.addLayer({ id: "site-fill", type: "fill", source: "sites", filter: ["==", ["geometry-type"], "Polygon"],
        paint: { "fill-color": LEVEL_EXPR, "fill-opacity": ["match", ["get", "level"], "high", 0.45, "medium", 0.35, 0.18] } });
      map.addLayer({ id: "site-line", type: "line", source: "sites",
        paint: { "line-color": LEVEL_EXPR, "line-width": ["match", ["get", "level"], "high", 2.5, "medium", 2, 1] } });
      map.addLayer({ id: "path", type: "line", source: "selected-path",
        layout: { "line-cap": "round", "line-join": "round" },
        paint: { "line-color": "#1b5fb3", "line-width": 3.5, "line-opacity": 0.85, "line-dasharray": [2, 1.2] } });
      // site markers at centroids so small sites are visible at any zoom
      map.addLayer({ id: "site-dot", type: "circle", source: "sites",
        filter: ["any", ["==", ["get", "level"], "high"], ["==", ["get", "level"], "medium"], ["==", ["get", "_marker"], true]],
        paint: { "circle-color": LEVEL_EXPR, "circle-radius": ["match", ["get", "level"], "high", 8, "medium", 6.5, 4.5],
          "circle-stroke-color": "#ffffff", "circle-stroke-width": 2 } });
      map.addLayer({ id: "exposed", type: "circle", source: "exposed",
        paint: { "circle-color": "#ffffff", "circle-radius": 5, "circle-stroke-color": "#1b5fb3", "circle-stroke-width": 2.5 } });
      map.addLayer({ id: "exposed-label", type: "symbol", source: "exposed",
        layout: { "text-field": ["get", "label"], "text-size": 12, "text-offset": [0, 1.1], "text-anchor": "top",
          "text-font": ["Open Sans Semibold"], "text-optional": true },
        paint: { "text-color": "#0b0b0b", "text-halo-color": "#ffffff", "text-halo-width": 1.6 } });
      // event source: a bullseye that reads on both basemaps
      map.addLayer({ id: "event-point-halo", type: "circle", source: "event-point",
        paint: { "circle-radius": 14, "circle-color": "rgba(0,0,0,0)", "circle-stroke-color": "#ffffff", "circle-stroke-width": 5 } });
      map.addLayer({ id: "event-point", type: "circle", source: "event-point",
        paint: { "circle-radius": 14, "circle-color": "rgba(0,0,0,0)", "circle-stroke-color": "#0b0b0b", "circle-stroke-width": 2.5 } });
      for (const id of ["site-fill", "site-dot"]) {
        map.on("mouseenter", id, () => (map.getCanvas().style.cursor = "pointer"));
        map.on("mouseleave", id, () => (map.getCanvas().style.cursor = ""));
      }
      ready.set(map, true);
      map.fire("himsat:ready");
    });
  }
  return map;
}

export function setData(map: MLMap, source: string, data: GeoJSON.FeatureCollection | GeoJSON.Feature): void {
  onReady(map, () => (map.getSource(source) as GeoJSONSource | undefined)?.setData(data));
}

export function setVisible(map: MLMap, layer: string, visible: boolean): void {
  onReady(map, () => map.setLayoutProperty(layer, "visibility", visible ? "visible" : "none"));
}

export function setBasemap(map: MLMap, which: "osm" | "s2"): void {
  setVisible(map, "osm", which === "osm");
  setVisible(map, "s2", which === "s2");
}

export function fitTo(map: MLMap, geoms: (GeoJSON.Geometry | null)[], maxZoom = 12): void {
  const b = new maplibregl.LngLatBounds();
  const add = (c: unknown): void => {
    if (Array.isArray(c) && typeof c[0] === "number") b.extend(c as [number, number]);
    else if (Array.isArray(c)) c.forEach(add);
  };
  for (const g of geoms) if (g && "coordinates" in g) add(g.coordinates);
  if (!b.isEmpty()) onReady(map, () => map.fitBounds(b, { padding: 60, maxZoom, duration: 600 }));
}
