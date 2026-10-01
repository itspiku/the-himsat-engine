import numpy as np
from shapely.geometry import box

from himsat.geo.geometry import from_geojson, mask_to_polygons, rasterize, to_geojson
from himsat.geo.grid import Grid, iter_tiles, utm_epsg


def test_utm_zone_for_nepal():
    assert utm_epsg(85.5, 28.3) == 32645
    assert utm_epsg(84.4, 28.5) == 32645
    assert utm_epsg(-70.0, -33.0) == 32719


def test_grid_from_bbox_is_snapped_and_roundtrips():
    g = Grid.from_lonlat_bbox((85.30, 28.12, 85.72, 28.42), 10)
    assert g.epsg == 32645
    assert g.x0 % 10 == 0 and g.y0 % 10 == 0
    r, c = g.lonlat_rowcol(85.5, 28.3)
    assert g.contains_rc(r, c)
    w, s, e, n = g.lonlat_bbox()
    assert w <= 85.30 and e >= 85.72 and s <= 28.12 and n >= 28.42


def test_tiles_cover_each_pixel_once():
    g = Grid(32645, 0, 10000, 10, 530, 470)
    count = np.zeros(g.shape, int)
    for t in iter_tiles(g, 200, 32):
        r0 = int(round((g.y0 - t.grid.y0) / g.res))
        c0 = int(round((t.grid.x0 - g.x0) / g.res))
        count[r0 + t.core_row0:r0 + t.core_row0 + t.core_height,
              c0 + t.core_col0:c0 + t.core_col0 + t.core_width] += 1
        assert t.grid.width <= 200 + 64 and t.grid.height <= 200 + 64
    assert (count == 1).all()


def test_mask_polygon_rasterize_roundtrip():
    g = Grid(32645, 300000, 3100000, 10, 100, 100)
    m = np.zeros(g.shape, bool)
    m[20:40, 30:70] = True
    polys = mask_to_polygons(m, g)
    assert len(polys) == 1 and abs(polys[0].area - 800 * 100) < 1e-6
    from himsat.geo.grid import reproject_geom

    back = rasterize([reproject_geom(polys[0], g.epsg, 4326)], g)
    assert abs(int(back.sum()) - 800) <= 40


def test_geojson_roundtrip():
    g = box(85.1, 28.1, 85.2, 28.2)
    assert from_geojson(to_geojson(g)).equals(g)
    assert to_geojson(None) is None
