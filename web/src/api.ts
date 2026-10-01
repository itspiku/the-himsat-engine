export interface SiteProps {
  code: string;
  kind: string;
  name: string;
  name_ne: string;
  lon: number;
  lat: number;
  elevation_m: number | null;
  level: string;
  score: number | null;
  assessed_at: string | null;
  area_m2?: number | null;
  exposed_assets?: number | null;
}

export interface Feature<P> {
  type: "Feature";
  geometry: GeoJSON.Geometry | null;
  properties: P;
}

export interface FC<P> {
  type: "FeatureCollection";
  features: Feature<P>[];
}

export interface Reason {
  code: string;
  value?: number;
  contribution?: number;
  text_en: string;
  text_ne: string;
}

export interface Exposed {
  kind: string;
  name: string;
  name_ne: string;
  lon: number;
  lat: number;
  travel_time_min: number;
  path_distance_km: number;
}

export interface SiteDetail extends SiteProps {
  geometry: GeoJSON.Geometry | null;
  flow_path: GeoJSON.LineString | null;
  attrs: Record<string, unknown>;
  risk: {
    level: string; score: number; hazard: number; exposure: number; confidence: number; assessed_at: string;
    reasons: Reason[]; potentially_dangerous_only: boolean;
  } | null;
  exposed: Exposed[];
  alerts: { uid: string; level: string; kind: string; issued_at: string; title_en: string; title_ne: string }[];
}

export interface Alert {
  uid: string;
  id: number;
  site_code: string;
  site_name: string;
  site_name_ne: string;
  site_kind: string;
  lon: number;
  lat: number;
  level: string;
  kind: string;
  status: string;
  issued_at: string;
  created_at: string;
  expires_at: string | null;
  title_en: string;
  title_ne: string;
  body_en: string;
  body_ne: string;
  sms_en: string;
  sms_ne: string;
  generator: string;
  reasons: Reason[];
  exposed: { name_en: string; name_ne: string; kind: string; travel_time_min: number; distance_km: number }[];
}

export interface Observation {
  at: string;
  kind: string;
  sensor: string;
  quality: number;
  values: Record<string, number>;
}

export interface Assessment {
  at: string;
  level: string;
  score: number;
  reasons: string[];
}

export interface HindcastSummary {
  name: string;
  aoi: string;
  aoi_name: string;
  period: [string, string];
  event_time: string | null;
  alerts: number;
}

export interface HindcastReport {
  aoi: { id: string; name: string; name_ne: string; bbox: [number, number, number, number] };
  period: [string, string];
  event_time: string | null;
  event_lonlat: [number, number] | null;
  scenes: { S1: number; S2: number; skipped: number; failed: number };
  sites: {
    code: string; name: string; name_ne: string; kind: string; lon: number; lat: number; elevation_m: number | null;
    distance_to_event_km: number | null; geometry: GeoJSON.Geometry | null; flow_path: GeoJSON.LineString | null;
    timeline: { at: string; level: string; score: number; reasons: string[] }[];
    observations: { at: string; kind: string; sensor: string; quality: number; values: Record<string, number> }[];
    exposure_top: { kind: string; name: string; name_ne: string; travel_time_min: number; path_distance_km: number }[];
    max_pre_event: { level: string; score: number; at: string; reasons: Reason[] } | null;
  }[];
  focus_sites: string[];
  alerts: {
    id: number; site: string; level: string; kind: string; evidence_at: string; available_at: string;
    lead_time_h: number | null; pre_event: boolean | null; title_en: string; title_ne: string; body_en: string;
    body_ne: string;
  }[];
  change_events: { kind: string; at: string; lon: number; lat: number; area_km2: number; confidence: number }[];
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function req<T>(path: string, init?: RequestInit & { key?: string }): Promise<T> {
  const headers: Record<string, string> = { Accept: "application/json" };
  if (init?.body) headers["Content-Type"] = "application/json";
  if (init?.key) headers["X-API-Key"] = init.key;
  const r = await fetch(path, { ...init, headers });
  if (!r.ok) {
    let msg = r.statusText;
    try {
      msg = (await r.json()).detail ?? msg;
    } catch {
      /* not json */
    }
    throw new ApiError(r.status, typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  if (r.status === 204) return undefined as T;
  return r.json() as Promise<T>;
}

export const api = {
  sites: (aoi?: string) => req<FC<SiteProps>>(`/api/sites${aoi ? `?aoi=${encodeURIComponent(aoi)}` : ""}`),
  site: (code: string) => req<SiteDetail>(`/api/sites/${encodeURIComponent(code)}`),
  observations: (code: string) => req<Observation[]>(`/api/sites/${encodeURIComponent(code)}/observations`),
  assessments: (code: string) => req<Assessment[]>(`/api/sites/${encodeURIComponent(code)}/assessments`),
  alerts: (active = true) => req<Alert[]>(`/api/alerts?active=${active}&days=90`),
  events: () => req<FC<{ kind: string; area_m2: number; detected_at: string; confidence: number }>>(`/api/events?days=60`),
  aois: () => req<{ id: string; name: string; name_ne: string; bbox: [number, number, number, number] }[]>(`/api/aois`),
  hindcasts: () => req<HindcastSummary[]>(`/api/hindcasts`),
  hindcast: (name: string) => req<HindcastReport>(`/api/hindcasts/${encodeURIComponent(name)}`),
  admin: {
    queue: (key: string) => req<Alert[]>(`/api/admin/alerts`, { key }),
    approve: (key: string, id: number, user: string) =>
      req<Record<string, number>>(`/api/admin/alerts/${id}/approve`, { method: "POST", key, body: JSON.stringify({ user }) }),
    cancel: (key: string, id: number, user: string) =>
      req<unknown>(`/api/admin/alerts/${id}/cancel`, { method: "POST", key, body: JSON.stringify({ user }) }),
    subscribers: (key: string) => req<Record<string, unknown>[]>(`/api/admin/subscribers`, { key }),
    addSubscriber: (key: string, body: Record<string, unknown>) =>
      req<unknown>(`/api/admin/subscribers`, { method: "POST", key, body: JSON.stringify(body) }),
    runs: (key: string) => req<Record<string, unknown>[]>(`/api/admin/runs`, { key }),
    run: (key: string, aoi_id: string) =>
      req<unknown>(`/api/admin/runs`, { method: "POST", key, body: JSON.stringify({ aoi_id }) }),
  },
};
