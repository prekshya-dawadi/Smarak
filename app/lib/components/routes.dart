import 'package:app/pages/book.dart';
import 'package:app/components/dashboard.dart';
import 'package:app/pages/homepage.dart';
import 'package:app/pages/map_map.dart';
import 'package:app/pages/quest.dart';
import 'package:app/pages/scan.dart';
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

final router = GoRouter(
  routes: [
    GoRoute(
      path: '/',
      builder: (context, state) => DashBoard(),
    ),
    GoRoute(path: '/map', builder: (context, state) => MapMap()),
    GoRoute(
      path: '/camera',
      builder: (context, state) => Scan(),
    ),
    GoRoute(
      path: '/book',
      builder: (context, state) => BookTicket(),
    ),
    GoRoute(
      path: '/quest',
      builder: (context, state) => Quest(),
    ),
  ],
);
