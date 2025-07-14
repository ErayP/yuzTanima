import 'package:flutter/material.dart';
import 'package:flutter_inappwebview/flutter_inappwebview.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: ThemeData(
        colorScheme: ColorScheme.light(
          primary: Colors.blue,
          secondary: Colors.grey.shade200,
        ),
        textTheme: TextTheme(
          bodyLarge: TextStyle(fontSize: 16, fontWeight: FontWeight.w400),
        ),
      ),
      home: VideoStreamPage(),
    );
  }
}

class VideoStreamPage extends StatelessWidget {
  bool _isDialogOpen = false;

  // Raspberry Pi’nizin wlan0 arayüzündeki IP’si
  final String baseUrl     = "http://172.20.10.2:5000";
  final String videoFeed   = "/video_feed";
  final String setStatus   = "/set_status";

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Dış Kapı Canlı Görüntüsü", style: TextStyle(fontWeight: FontWeight.bold)),
        backgroundColor: Colors.transparent,
        elevation: 0,
        centerTitle: true,
      ),
      body: Column(
        children: [
          // Video Stream Area
          Expanded(
            child: InAppWebView(
              initialUrlRequest: URLRequest(
                url: WebUri(baseUrl + videoFeed),
              ),
              initialOptions: InAppWebViewGroupOptions(
                crossPlatform: InAppWebViewOptions(
                  useOnDownloadStart: true,
                  mediaPlaybackRequiresUserGesture: false,
                ),
              ),
            ),
          ),
          // Buttons Row
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 20, horizontal: 20),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                _buildIconButton(
                  context,
                  backgroundColor: Colors.white,
                  iconColor: Colors.green,
                  icon: Icons.lock_open,
                  onPressed: () async {
                    if (!_isDialogOpen) {
                      _isDialogOpen = true;
                      await _sendStatusToPython(1);
                      _showPopup(context, "Kapı Açıldı!");
                    }
                  },
                ),
                _buildIconButton(
                  context,
                  backgroundColor: Colors.white,
                  iconColor: Colors.red,
                  icon: Icons.lock,
                  onPressed: () async {
                    if (!_isDialogOpen) {
                      _isDialogOpen = true;
                      await _sendStatusToPython(0);
                      _showPopup(context, "Kapı Kapandı!");
                    }
                  },
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildIconButton(BuildContext context, {
    required Color backgroundColor,
    required Color iconColor,
    required IconData icon,
    required VoidCallback onPressed,
  }) {
    return Center(
      child: SizedBox(
        width: 70,
        height: 70,
        child: ElevatedButton(
          style: ElevatedButton.styleFrom(
            backgroundColor: backgroundColor,
            shape: CircleBorder(),
            padding: EdgeInsets.all(10),
          ),
          onPressed: onPressed,
          child: Icon(icon, size: 36, color: iconColor),
        ),
      ),
    );
  }

  Future<void> _sendStatusToPython(int status) async {
    final url = Uri.parse(baseUrl + setStatus);
    try {
      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'status': status}),
      );

      if (response.statusCode == 200) {
        print("Durum başarıyla gönderildi: $status");
      } else {
        print("Hata: ${response.statusCode}");
      }
    } catch (e) {
      print("Bağlantı hatası: $e");
    }
  }

  void _showPopup(BuildContext context, String message) {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return Align(
          alignment: Alignment(0.0, 0.8),
          child: Material(
            color: Colors.transparent,
            child: AlertDialog(
              title: Text("Bilgilendirme"),
              content: Text(message),
            ),
          ),
        );
      },
    ).then((_) {
      _isDialogOpen = false;
    });

    Future.delayed(Duration(seconds: 2), () {
      if (Navigator.of(context).canPop()) {
        Navigator.of(context).pop();
      }
    });
  }
}
