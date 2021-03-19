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
    landsat5_sr = ee.ImageCollection("LANDSAT/LT05/C01/T1_SR"),
    landsat7_sr = ee.ImageCollection("LANDSAT/LE07/C01/T1_SR"),
    landsat8_sr = ee.ImageCollection("LANDSAT/LC08/C01/T1_SR"),
    imageVisParam = {"opacity":1,"bands":["B1","B2","B3"],"gamma":1},
    bbox = 
    /* color: #d63000 */
    /* displayProperties: [
      {
        "type": "rectangle"
      }
    ] */
    ee.Geometry.Polygon(
        [[[28.266653672292577, 52.48438593323408],
          [28.266653672292577, 50.501129231261004],
          [31.474661484792577, 50.501129231261004],
          [31.474661484792577, 52.48438593323408]]], null, false);


// Filter for spatial and temporal regions of interest
var spatialFiltered = landsat8_sr.filterBounds(bbox);
print('spatialFiltered', spatialFiltered);

var temporalFiltered_JFM = spatialFiltered.filterDate('2014-01-01', '2014-03-31');
print('temporalFiltered', temporalFiltered_JFM);

var temporalFiltered_AMJ = spatialFiltered.filterDate('2014-04-01', '2014-06-30');
print('temporalFiltered', temporalFiltered_AMJ);

var temporalFiltered_JAS = spatialFiltered.filterDate('2014-07-01', '2014-09-30');
print('temporalFiltered', temporalFiltered_JAS);

var temporalFiltered_OND = spatialFiltered.filterDate('2014-10-01', '2014-12-31');
print('temporalFiltered', temporalFiltered_OND);

/**
 * Function to mask clouds based on the pixel_qa band of Landsat 8 SR data.
 * @param {ee.Image} image input Landsat 8 SR image
 * @return {ee.Image} cloudmasked Landsat 8 image
 */
function maskL8sr(image) {
  // Bits 3 and 5 are cloud shadow and cloud, respectively.
  var cloudShadowBitMask = (1 << 3);
  var cloudsBitMask = (1 << 5);
  // Get the pixel QA band.
  var qa = image.select('pixel_qa');
  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0)
                 .and(qa.bitwiseAnd(cloudsBitMask).eq(0));
  return image.updateMask(mask);
}

//Apply the cloud mask to relevant data

var dataset_JFM = temporalFiltered_JFM.map(maskL8sr);

var dataset_AMJ = temporalFiltered_AMJ.map(maskL8sr);

var dataset_JAS = temporalFiltered_JAS.map(maskL8sr);

var dataset_OND = temporalFiltered_OND.map(maskL8sr);

Map.centerObject(bbox, 9);

var visParams = {
  bands: ['B4', 'B3','B2'],
  min: 0,
  max: 3000,
  gamma: 1.4,
};

//Take median composites of visible and infrared bands
var image_JFM = dataset_JFM.median().select(['B1', 'B3', 'B2','B4','B5','B6','B7']);
var image_AMJ = dataset_AMJ.median().select(['B1', 'B3', 'B2','B4','B5','B6','B7']);
var image_JAS = dataset_JAS.median().select(['B1', 'B3', 'B2','B4','B5','B6','B7']);
var image_OND = dataset_OND.median().select(['B1', 'B3', 'B2','B4','B5','B6','B7']);

//Display the visible bands of a composite image
Map.addLayer(dataset_JFM.median(), visParams);


//Export the image, specifying scale and region.
Export.image.toDrive({
  image: image_JFM,
  description: 'L8_chern_2014_JFM',
  scale: 30,
  region: bbox,
  fileFormat: "GeoTIFF",
  formatOptions: {
    cloudOptimized: true
  }
});

//Export the image, specifying scale and region.
Export.image.toDrive({
  image: image_JFM,
  description: 'L8_hab_2014_JFM',
  scale: 30,
  region: habitat_bbox,
  fileFormat: "GeoTIFF",
  formatOptions: {
    cloudOptimized: true
  }
});

//Export the image, specifying scale and region.
Export.image.toDrive({
  image: image_AMJ,
  description: 'L8_chern_2014_AMJ',
  scale: 30,
  region: bbox,
  fileFormat: "GeoTIFF",
  formatOptions: {
    cloudOptimized: true
  }
});

//Export the image, specifying scale and region.
Export.image.toDrive({
  image: image_AMJ,
  description: 'L8_hab_2014_AMJ',
  scale: 30,
  region: habitat_bbox,
  fileFormat: "GeoTIFF",
  formatOptions: {
    cloudOptimized: true
  }
});

//Export the image, specifying scale and region.
Export.image.toDrive({
  image: image_JAS,
  description: 'L8_chern_2014_JAS',
  scale: 30,
  region: bbox,
  fileFormat: "GeoTIFF",
  formatOptions: {
    cloudOptimized: true
  }
});

//Export the image, specifying scale and region.
Export.image.toDrive({
  image: image_JAS,
  description: 'L8_hab_2014_JAS',
  scale: 30,
  region: habitat_bbox,
  fileFormat: "GeoTIFF",
  formatOptions: {
    cloudOptimized: true
  }
});

//Export the image, specifying scale and region.
Export.image.toDrive({
  image: image_OND,
  description: 'L8_chern_2014_OND',
  scale: 30,
  region: bbox,
  fileFormat: "GeoTIFF",
  formatOptions: {
    cloudOptimized: true
  }
});

//Export the image, specifying scale and region.
Export.image.toDrive({
  image: image_OND,
  description: 'L8_hab_2014_OND',
  scale: 30,
  region: habitat_bbox,
  fileFormat: "GeoTIFF",
  formatOptions: {
    cloudOptimized: true
  }
});