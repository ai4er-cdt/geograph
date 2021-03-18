var nuclear_plant = /* color: #d63000 */ee.Geometry.Point([30.10431351791908, 51.390526457717954]),
    sentinel2_toa = ee.ImageCollection("COPERNICUS/S2"),
    habitat_bbox = 
    /* color: #98ff00 */
    /* displayProperties: [
      {
        "type": "rectangle"
      }
    ] */
    ee.Geometry.Polygon(
        [[[26.88698982066135, 52.46476723608711],
          [26.88698982066135, 51.57234663784468],
          [28.60085700816135, 51.57234663784468],
          [28.60085700816135, 52.46476723608711]]], null, false),
    imageVisParam = {"opacity":1,"bands":["B1","B2","B3"],"gamma":1},
    bbox = 
    /* color: #d63000 */
    /* shown: false */
    /* displayProperties: [
      {
        "type": "rectangle"
      }
    ] */
    ee.Geometry.Polygon(
        [[[28.266653672292577, 52.48438593323408],
          [28.266653672292577, 50.501129231261004],
          [31.474661484792577, 50.501129231261004],
          [31.474661484792577, 52.48438593323408]]], null, false),
    sentinel2_sr = ee.ImageCollection("COPERNICUS/S2_SR"),
    s2Clouds = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY"),
    polesia_bbox = 
    /* color: #e21fff */
    /* displayProperties: [
      {
        "type": "rectangle"
      }
    ] */
    ee.Geometry.Polygon(
        [[[27.08183119, 52.20281607],
          [27.08183119, 51.46442232],
          [28.71064848, 51.46442232],
          [28.71064848, 52.20281607]]], null, false);

var START_DATE = ee.Date('2017-10-01');
var END_DATE = ee.Date('2017-12-31');
var MAX_CLOUD_PROBABILITY = 65;
var UTM35N = 'EPSG:32635';

//Function to mask clouds
function maskClouds(img) {
    var clouds = ee.Image(img.get('cloud_mask')).select('probability');
    var isNotCloud = clouds.lt(MAX_CLOUD_PROBABILITY);
    return img.updateMask(isNotCloud);
}
          
// The masks for the 10m bands sometimes do not exclude bad data at
// scene edges, so we apply masks from the 20m bands as well.
function maskEdges(s2_img) {
    return s2_img.updateMask(
        s2_img.select('B8A').mask());
}
          
// Filter input collections by desired data range and region. Select visible, infrared and red edge bands
var criteria = ee.Filter.and(
    ee.Filter.bounds(polesia_bbox), ee.Filter.date(START_DATE, END_DATE));
sentinel2_sr = sentinel2_sr.filter(criteria).select(['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']).map(maskEdges);
s2Clouds = s2Clouds.filter(criteria);
          
// Join S2 SR with cloud probability dataset to add cloud mask.
var s2SrWithCloudMask = ee.Join.saveFirst('cloud_mask').apply({
    primary: sentinel2_sr,
    secondary: s2Clouds,
    condition:
        ee.Filter.equals({leftField: 'system:index', rightField: 'system:index'})
});

//Take median composite 
var s2CloudMasked =
    ee.ImageCollection(s2SrWithCloudMask).map(maskClouds).median();
              
var rgbVis = {min: 0, max: 3000, bands: ['B4', 'B3', 'B2']};

//Visualise image in viewer
Map.addLayer(
    s2CloudMasked, rgbVis, 'S2 SR masked at ' + MAX_CLOUD_PROBABILITY + '%',
    true);
          
          
          
//Export the image, specifying scale and region.
Export.image.toDrive({
    image: s2CloudMasked,
    description: 'S2_pol_2017_OND',
    scale: 10,
    region: polesia_bbox,
    crs: 'EPSG:32635',
    fileFormat: "GeoTIFF",
    maxPixels: 1e10
    });