import 'package:app/pages/book.dart';
import 'package:app/pages/homepage.dart';
import 'package:app/pages/map_map.dart';
import 'package:app/pages/quest.dart';
import 'package:app/pages/scan.dart';
import 'package:flutter/material.dart';

class DashBoard extends StatefulWidget {
  const DashBoard({super.key});

  @override
  State<DashBoard> createState() => _DashBoardState();
}

class _DashBoardState extends State<DashBoard> {
  int currentIndex = 0;
  // final Function(int) onTap;
  final List<Widget> widgets = [
    HomePage(),
    MapMap(),
    Scan(),
    BookTicket(),
    Quest(),
  ];
  @override
  Widget build(BuildContext context) {
    return Scaffold(
        body: widgets[currentIndex],
        bottomNavigationBar: BottomNavigationBar(
          // backgroundColor: Color(0xFFF3F8F9),
          backgroundColor: Colors.transparent,
          selectedItemColor: Color.fromARGB(255, 96, 92, 92),
          unselectedItemColor: Colors.grey,
          currentIndex: currentIndex,
          onTap: (int index) {
            setState(() {
              currentIndex = index;
            });
          },
          items: const <BottomNavigationBarItem>[
            BottomNavigationBarItem(
              icon: Icon(Icons.home),
              label: 'Home',
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.map_outlined),
              label: 'Map',
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.camera),
              label: 'AR View',
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.airplane_ticket_outlined),
              label: 'Book',
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.directions_walk_outlined),
              label: 'Quest',
            ),
          ],
        ));
  }
}
