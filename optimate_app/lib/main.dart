import 'package:flutter/material.dart';
import 'landing_page.dart';

void main() {
  runApp(const OptifocusLiteApp());
}

class OptifocusLiteApp extends StatelessWidget {
  const OptifocusLiteApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Optifocus Lite',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        textTheme: ThemeData.light().textTheme,
        primarySwatch: Colors.amber,
        scaffoldBackgroundColor: Colors.white,
      ),
      home: const SplashScreen(),
    );
  }
}
