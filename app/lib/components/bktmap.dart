import 'package:galli_map/galli_map.dart';
import 'package:flutter/material.dart';

final GalliController controller = GalliController(
  authKey: "25fb86bf-71cb-4dd2-8ec2-f9792d8a327a	",
        zoom: 16,


);

@override
Widget build(BuildContext context) {
  return Scaffold(
    appBar: AppBar(
      title: Text("Map"),
    ),
    body: GalliMap(
      // authKey: "25fb86bf-71cb-4dd2-8ec2-f9792d8a327a	",
      controller: controller,
      onTap: (tap) {},
      circles: [
        GalliCircle(
            center: LatLng(27.12441, 67.12412),
            radius: 32,
            color: Colors.white,
            borderStroke: 3,
            borderColor: Colors.black)
      ],
      lines: [
        GalliLine(
            line: [
              LatLng(27.12441, 67.12412),
              LatLng(27.12441, 67.12412),
              LatLng(27.12441, 67.12412),
              LatLng(27.12441, 67.12412)
            ],
            borderColor: Colors.blue,
            borderStroke: 1,
            lineColor: Colors.white,
            lineStroke: 2)
      ],
      polygons: [
        GalliPolygon(polygon: [
          LatLng(27.12441, 67.12412),
          LatLng(27.12441, 67.12412),
          LatLng(27.12441, 67.12412),
          LatLng(27.12441, 67.12412)
        ], borderColor: Colors.red, borderStroke: 2, color: Colors.green),
      ],
      markers: [
        GalliMarker(
            latlng: LatLng(27.12441, 67.12412),
            markerWidget: const Icon(Icons.location_city))
      ],
    ),
  );
}
