import 'dart:io';

import 'package:app/components/bottom_navigation.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/widgets.dart';
import 'package:galli_map/galli_map.dart';
import 'package:go_router/go_router.dart';

final GalliController controller = GalliController(
  authKey: "25fb86bf-71cb-4dd2-8ec2-f9792d8a327a",
  zoom: 16,
  maxZoom: 22.0,
  initialPosition: LatLng(27.671144, 85.429205),
);
final GalliMethods galliMethods =
    GalliMethods("25fb86bf-71cb-4dd2-8ec2-f9792d8a327a	");
final List<String> yourList = [
  'golden.jpg',
  'golden.jpg',
  'golden.jpg',
  'person.jpg',
  'person.jpg',
];

enum Location { Bhaktapur, Kathmandu, Lalitpur }

class HomePage extends StatefulWidget {
  const HomePage({Key? key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final TextEditingController locationController = TextEditingController();

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Column(
        // mainAxisAlignment: MainAxisAlignment.spaceAround,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          DropdownMenu<Location>(
            width: MediaQuery.of(context).size.width * 0.8,
            inputDecorationTheme: InputDecorationTheme(
              hoverColor: Colors.grey,
              disabledBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(10),
              ),
            ),
            controller: TextEditingController(),
            leadingIcon: Icon(Icons.location_city),
            label: const Text('Location'),
            dropdownMenuEntries:
                Location.values.map<DropdownMenuEntry<Location>>(
              (Location location) {
                return DropdownMenuEntry<Location>(
                  label: "${location.toString().split('.').last}",
                  value: location,
                );
              },
            ).toList(),
          ),
          Text(
            'Welcome to AR Bhaktapur Durbar Square ',
            style: TextStyle(fontSize: 25, fontWeight: FontWeight.bold),
            textAlign: TextAlign.left,
          ),
          Container(
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              // crossAxisAlignment: CrossAxisAlignment.start,
              children: const [
                Text(
                  "Nearby",
                  style: TextStyle(fontSize: 20, fontWeight: FontWeight.w600),
                ),
                Icon(Icons.keyboard_arrow_right)
              ],
            ),
          ),
          Container(
            height: 350,
            child: ListView.builder(
              scrollDirection: Axis.horizontal,
              padding: EdgeInsets.all(20.0),
              shrinkWrap: true,
              itemCount: 5,
              itemBuilder: (context, index) => Container(
                // height: 200,
                width: 200.0,
                child: Stack(
                    fit: StackFit.loose,
                    alignment: Alignment.bottomCenter,
                    children: [
                      Image.asset('assets/${yourList[index]}'),
                      Card(
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(10.0),
                        ),
                        color: Colors.white,
                        // child: Padding(
                        //   padding: const EdgeInsets.all(8.0),
                        child: Container(
                          height: 100,
                          child: Column(
                            children: [
                              Row(
                                children: [
                                  Text('Bhaktapur Durbar Square'),
                                  Spacer(),
                                  Icon(Icons.favorite_border)
                                ],
                              ),
                              Row(
                                children: [
                                  Text('Bhaktapur, Nepal'),
                                  Spacer(),
                                  Text('1900'),
                                ],
                              ),
                            ],
                          ),
                        ),
                      ),
                    ]),
              ),
            ),
          ),
          Container(
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              // crossAxisAlignment: CrossAxisAlignment.start,
              children: const [
                Text(
                  "Nearby",
                  style: TextStyle(fontSize: 20, fontWeight: FontWeight.w600),
                ),
                Icon(Icons.keyboard_arrow_right)
              ],
            ),
          ),
        ],
      ),
    );
  }
}
