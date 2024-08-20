import xarray as xr
from jcclass import jc

min_lon, max_lon, min_lat, max_lat = (4, 32, 54, 72)  # nordic

da = xr.open_dataset("sample_data/era5_hourly_highres.nc")

# Subset on limited area 'nordic'
da_subset = da.where(
    (da.longitude >= min_lon) & (da.longitude <= max_lon) & (da.latitude >= min_lat) & (da.latitude <= max_lat),
    drop=True,
)


# Original test data
def test_global_data():
    jc_res = jc(da)
    cts_27 = jc_res.classification()
    jc_res.eleven_cts(cts_27)


# Limited area 'nordic' test data
def test_limited_area_data():
    jc_res_nordic = jc(da_subset)
    cts_27 = jc_res_nordic.classification()
    jc_res_nordic.eleven_cts(cts_27)
