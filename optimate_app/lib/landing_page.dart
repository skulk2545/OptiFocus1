import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'camera_screen.dart';

class SplashScreen extends StatefulWidget {
const SplashScreen({super.key});

@override
State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
@override
void initState() {
super.initState();
Future.delayed(const Duration(seconds: 2), () {
if (mounted) {
Navigator.pushReplacement(
context,
MaterialPageRoute(builder: (_) => const StartScreen()),
);
}
});
}

@override
Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            const Spacer(),
            Image.asset(
              'assets/images/OPTIFOCUS_LOGO.png',
              width: 200,
              fit: BoxFit.contain,
            ),
            const SizedBox(height: 18),
            const Text(
              'Optifocus',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
            Expanded(
              child: Align(
                alignment: Alignment.bottomCenter,
                child: Padding(
                  padding: const EdgeInsets.only(bottom: 10.0),
                  child: const Text(
                    '© copyright 2025 Optifocus Pvt. Ltd.',
                    style: TextStyle(fontSize: 16, color: Color.fromARGB(255, 15, 15, 15)),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
}
}

class StartScreen extends StatelessWidget {
const StartScreen({super.key});

@override
Widget build(BuildContext context) {
return Scaffold(
body: SafeArea(
child: Center(
child: Column(
mainAxisSize: MainAxisSize.min,
children: [
Image.asset(
'assets/images/OPTIFOCUS_LOGO.png',
width: 160,
fit: BoxFit.contain,
),
const SizedBox(height: 28),
ElevatedButton(
onPressed: () {
Navigator.push(
context,
MaterialPageRoute(builder: (_) => const ConsentScreen()),
);
},
style: ElevatedButton.styleFrom(
backgroundColor: Colors.amber,
padding:
const EdgeInsets.symmetric(horizontal: 36, vertical: 12),
shape: RoundedRectangleBorder(
borderRadius: BorderRadius.circular(10),
),
),
child: const Text(
'START',
style: TextStyle(
color: Colors.black,
fontSize: 16,
fontWeight: FontWeight.w700),
),

),
          const SizedBox(height: 12),
          const Text(
            '© copyright 2025 Optifocus Pvt. Ltd.',
            style: TextStyle(fontSize: 14, color: Color.fromARGB(255, 15, 15, 15)),
          ),
],
),
),
),
);
}
}

class ConsentScreen extends StatefulWidget {
const ConsentScreen({super.key});

@override
State<ConsentScreen> createState() => _ConsentScreenState();
}

class _ConsentScreenState extends State<ConsentScreen> {
bool isExpanded = false;

final String introText = '''
User Consent
To the End User,

By Clicking on the “Accept” button, we confirm that subscriber of this App has taken consent from YOU - the End user /Customer /Client / Patient / or by whatever relation (henceforth called Customer), whether verbally / in writing or any other format before proceeding. The Subscriber (henceforth called User) has conveyed that the Customer(s) have agreed to proceed and process their biometric data with our tool. This allows the User to capture Customers’ facial image for taking accurate measurements through their mobile devices for optimum results.
The biometric data will be processed for delivering results of highest quality and will be stored securely on our systems and will only be accessed by authorized personnel.
By giving your consent, you (Customer) agree to the digital processing of your facial image and confirm that you (Customer) are at least 18 years old, meet the digital age of consent applicable in your country or have parental/guardian consent to have your data processed.
By continuing, you (Customer) accept our Privacy Policy terms and the processing of your biometric data.
''';

final String expandedText = '''
Details of Processing
Data Controller:
Optifocus Pvt. Ltd.

Purpose of Processing:
Processing your biometric data (eye centre and frame dimensions) through image processing for visualizing the exact point where the pupil centre lies with respect to the new frame. Your data will be used only for this visualization / calculation and for no other purposes.

Data Processed:
• Eye Centre, Frame Dimensions
• Biometric data derived from facial image analysis

Legal Basis for Processing:
As per applicable data protection laws, biometric data processing is based on your explicit consent.

Data Sharing:
Your biometric data will not be shared with any third party and will be stored securely on our systems. It is processed only for the simulation/measurement purpose.

Data Retention Period:
Your biometric data is processed through imaging technique and will not be retained for more than 30 days after the process is complete.

Your Rights:
You (Customer) have the right to withdraw your consent at any time. Withdrawing consent will not affect the legality of processing carried out prior to withdrawal.
Since we do not retain your facial data for more than 30 days after the session, rights to access, correct, delete, or restrict processing do not apply after the retention period ends. You (Customer) can exercise your right to access, correct, delete, or restrict processing before 30th day from the session.

Age Declaration:
I declare that I am at least 18 years old, or meet the digital age of consent in my country, or have parental/guardian consent to have my data processed.

Contact Information:
To exercise your rights or withdraw consent, please contact us at:
[consultants@optifocus.com](mailto:consultants@optifocus.com)

Withdrawing Consent:
You (Customer) may withdraw your consent at any time without providing a reason. This will not affect the legality of processing carried out before withdrawal.
''';

@override
Widget build(BuildContext context) {
final TextStyle headingStyle =
const TextStyle(fontSize: 16.5, fontWeight: FontWeight.w700);
final TextStyle bodyStyle = const TextStyle(fontSize: 14.5, height: 1.45);


return Scaffold(
  appBar: AppBar(
    title: const Text('User Consent'),
    backgroundColor: Colors.amber,
    foregroundColor: Colors.black,
  ),
  body: Padding(
    padding: const EdgeInsets.all(14.0),
    child: Column(
      children: [
        Expanded(
          child: SingleChildScrollView(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(introText, style: bodyStyle),
                const SizedBox(height: 8),
                const Divider(),
                if (isExpanded) ...[
                  const SizedBox(height: 8),
                  Text('Details of Processing', style: headingStyle),
                  const SizedBox(height: 8),
                  Text(expandedText, style: bodyStyle),
                ],
              ],
            ),
          ),
        ),
        Align(
          alignment: Alignment.center,
          child: TextButton(
            onPressed: () => setState(() => isExpanded = !isExpanded),
            child: Text(
              isExpanded ? 'Read Less ▲' : 'Read More ▼',
              style: const TextStyle(
                  color: Colors.blueAccent, fontWeight: FontWeight.w700),
            ),
          ),
        ),
        const SizedBox(height: 8),
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            ElevatedButton(
              onPressed: () {
                Navigator.pushAndRemoveUntil(
                  context,
                  MaterialPageRoute(builder: (_) => const SplashScreen()),
                  (route) => false,
                );
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.redAccent,
                padding: const EdgeInsets.symmetric(
                    horizontal: 30, vertical: 12),
              ),
              child:
                  const Text('DECLINE', style: TextStyle(fontSize: 14)),
            ),
            ElevatedButton(
              onPressed: () async {
                // Fetch available cameras first
                List<CameraDescription> cameras = await availableCameras();
                if (!mounted) return;

                Navigator.pushReplacement(
                  context,
                  MaterialPageRoute(
                    builder: (context) =>
                        CameraScreen(cameras: cameras),
                  ),
                );
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.amber,
                padding: const EdgeInsets.symmetric(
                    horizontal: 30, vertical: 12),
              ),
              child: const Text(
                'ACCEPT',
                style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w700,
                    color: Colors.black),
              ),
            ),
          ],
        ),
        const SizedBox(height: 10),
      ],
    ),
  ),
);


}
}
