import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

class Quest extends StatelessWidget {
  const Quest({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Quest Page'),
      ),
      body: Column(
        children: [
          ElevatedButton(
            onPressed: () {
              // Navigate back to the homepage
              context.go('/'); // Assuming the homepage route is '/'
            },
            child: Text('Go back to Homepage'),
          ),
          Center(
            child: Text(
              'Welcome to the Quest Page!',
              style: TextStyle(fontSize: 24),
            ),
          ),
        ],
      ),
    );
  }
}
