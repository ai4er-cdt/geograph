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
var spatialFiltered = landsat5_sr.filterBounds(bbox);
print('spatialFiltered', spatialFiltered);

var temporalFiltered_JFM = spatialFiltered.filterDate('1984-01-01', '1984-03-31');
print('temporalFiltered', temporalFiltered_JFM);

var temporalFiltered_AMJ = spatialFiltered.filterDate('1984-04-01', '1984-06-30');
print('temporalFiltered', temporalFiltered_AMJ);

var temporalFiltered_JAS = spatialFiltered.filterDate('1984-07-01', '1984-09-30');
print('temporalFiltered', temporalFiltered_JAS);

var temporalFiltered_OND = spatialFiltered.filterDate('1984-10-01', '1984-12-31');
print('temporalFiltered', temporalFiltered_OND);

/**
 * Function to mask clouds based on the pixel_qa band of Landsat SR data.
 * @param {ee.Image} image Input Landsat SR image
 * @return {ee.Image} Cloudmasked Landsat image
 */
var cloudMaskL457 = function(image) {
  var qa = image.select('pixel_qa');
  // If the cloud bit (5) is set and the cloud confidence (7) is high
  // or the cloud shadow bit is set (3), then it's a bad pixel.
  var cloud = qa.bitwiseAnd(1 << 5)
                  .and(qa.bitwiseAnd(1 << 7))
                  .or(qa.bitwiseAnd(1 << 3));
  // Remove edge pixels that don't occur in all bands
  var mask2 = image.mask().reduce(ee.Reducer.min());
  return image.updateMask(cloud.not()).updateMask(mask2);
};

//Apply the cloud mask to relevant data

var dataset_JFM = temporalFiltered_JFM.map(cloudMaskL457);

var dataset_AMJ = temporalFiltered_AMJ.map(cloudMaskL457);

var dataset_JAS = temporalFiltered_JAS.map(cloudMaskL457);

var dataset_OND = temporalFiltered_OND.map(cloudMaskL457);

Map.centerObject(bbox, 9);

var visParams = {
  bands: ['B3', 'B2', 'B1'],
  min: 0,
  max: 3000,
  gamma: 1.4,
};

//Take median composites of visible and infrared bands
var image_JFM = dataset_JFM.median().select(['B3', 'B2', 'B1','B4','B5','B7']);
var image_AMJ = dataset_AMJ.median().select(['B3', 'B2', 'B1','B4','B5','B7']);
var image_JAS = dataset_JAS.median().select(['B3', 'B2', 'B1','B4','B5','B7']);
var image_OND = dataset_OND.median().select(['B3', 'B2', 'B1','B4','B5','B7']);

//Display the visible bands of a composite image
Map.addLayer(dataset_AMJ.median(), visParams);

//Export the image, specifying scale and region.
Export.image.toDrive({
  image: image_JFM,
  description: 'L5_chern_1984_JFM',
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
  description: 'L5_hab_1984_JFM',
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
  description: 'L5_chern_1984_AMJ',
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
  description: 'L5_hab_1984_AMJ',
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
  description: 'L5_chern_1984_JAS',
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
  description: 'L5_hab_1984_JAS',
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
  description: 'L5_chern_1984_OND',
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
  description: 'L5_hab_1984_OND',
  scale: 30,
  region: habitat_bbox,
  fileFormat: "GeoTIFF",
  formatOptions: {
    cloudOptimized: true
  }
});
