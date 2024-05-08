import 'package:app/components/bottom_navigation.dart';
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

class Scan extends StatelessWidget {
  const Scan({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Scan Page'),
      ),
      body: Column(children: [
        Text(
          'Welcome to the Scan Page!',
          style: TextStyle(fontSize: 24),
        ),
        ElevatedButton(
          onPressed: () {
            // Navigate back to the homepage
            context.go('/'); // Assuming the homepage route is '/'
          },
          child: Text('Go back to Homepage'),
        ),
      ]),
    );
  }
}
