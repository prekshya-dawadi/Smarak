import 'package:app/components/bottom_navigation.dart';
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

class BookTicket extends StatelessWidget {
  const BookTicket({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        body: Column(
      children: [
        SizedBox(
          height: 50,
        ),
        ElevatedButton(
          onPressed: () {
            // Navigate back to the homepage
            context.go('/'); // Assuming the homepage route is '/'
          },
          child: Text('Go back to Homepage'),
        ),
        Center(
          child: Text("Book Ticket"),
        ),
      ],
    ));
  }
}
