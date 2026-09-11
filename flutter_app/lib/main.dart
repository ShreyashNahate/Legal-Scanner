// Legal Metrology Compliance Scanner - complete Flutter front-end.
//
// Flow: pick/take a photo -> upload to backend -> backend does
// OCR -> field extraction -> Legal Metrology RULE VALIDATION
// (PASS/FAIL/REVIEW) -> nutrition/claims analysis -> this screen
// displays ALL of it: product info, compliance verdicts, and
// nutrition facts.
//
// CONNECTING TO YOUR BACKEND:
// "localhost" from a phone does NOT mean your computer.
//   - Physical phone (same Wi-Fi/hotspot as your computer) -> use your
//     computer's LAN IP (from `ifconfig`/`ipconfig`), e.g. 10.154.55.8
//   - Android emulator -> 10.0.2.2
//   - iOS simulator -> localhost
// Set this in baseUrl below.

import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:legal_metrology_scanner_app/history_page.dart';

void main() {
  runApp(const ComplianceScannerApp());
}

class ComplianceScannerApp extends StatelessWidget {
  const ComplianceScannerApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Legal Metrology Scanner',
      theme: ThemeData(primarySwatch: Colors.indigo, useMaterial3: true),
      home: const ScannerHomePage(),
    );
  }
}

class ScannerHomePage extends StatefulWidget {
  const ScannerHomePage({super.key});

  @override
  State<ScannerHomePage> createState() => _ScannerHomePageState();
}

class _ScannerHomePageState extends State<ScannerHomePage> {
  // ---- UPDATE THIS to match your backend's address ----
  static const String baseUrl = 'http://172.19.210.8:8000';

  final ImagePicker _picker = ImagePicker();

  File? _selectedImage;
  bool _isLoading = false;
  String _scanMethod = 'hybrid'; // 'hybrid' | 'ai-vision' | 'ai-text' | 'regex'
  String? _errorMessage;
  Map<String, dynamic>? _result;

  Future<ImageSource?> _pickSourceSheet() {
    return showModalBottomSheet<ImageSource>(
      context: context,
      builder: (context) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            ListTile(
              leading: const Icon(Icons.camera_alt),
              title: const Text('Take Photo'),
              onTap: () => Navigator.pop(context, ImageSource.camera),
            ),
            ListTile(
              leading: const Icon(Icons.photo_library),
              title: const Text('From Gallery'),
              onTap: () => Navigator.pop(context, ImageSource.gallery),
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _saveToHistory() async {
    if (_result == null) return;
    try {
      final uri = Uri.parse('$baseUrl/save-scan');
      final res = await http.post(
        uri,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'product': _result!['product'],
          'report': _result!['report'],
          'nutrition_analysis': _result!['nutrition_analysis'],
          'scan_method': _scanMethod,
        }),
      );
      if (res.statusCode == 200 && mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Saved to history')),
        );
      } else if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Save failed (${res.statusCode})')),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Could not reach backend: $e')),
        );
      }
    }
  }

  Future<void> _showOtherScanOptions() async {
    final choice = await showModalBottomSheet<String>(
      context: context,
      builder: (context) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            ListTile(
              leading: const Icon(Icons.remove_red_eye),
              title: const Text('AI Vision Only'),
              subtitle: const Text(
                  'AI reads the image directly, no OCR/regex fallback'),
              onTap: () => Navigator.pop(context, 'ai-vision'),
            ),
            ListTile(
              leading: const Icon(Icons.auto_awesome),
              title: const Text('AI + OCR Scan'),
              subtitle: const Text(
                  'OCR reads text, then AI cleans/structures it (faster than pure AI vision)'),
              onTap: () => Navigator.pop(context, 'ai-text'),
            ),
            ListTile(
              leading: const Icon(Icons.text_snippet),
              title: const Text('Basic OCR Scan (no AI)'),
              subtitle: const Text(
                  'Regex-based, fastest, no API key needed, least accurate on messy photos'),
              onTap: () => Navigator.pop(context, 'regex'),
            ),
          ],
        ),
      ),
    );
    if (choice == null) return;
    final source = await _pickSourceSheet();
    if (source != null) await _pickImage(source, method: choice);
  }

  Future<void> _pickImage(ImageSource source,
      {String method = 'hybrid'}) async {
    // No imageQuality param on purpose - image_picker's internal
    // downscaling can return a path before the file finishes writing
    // on some Android devices, producing a corrupt upload.
    final XFile? picked = await _picker.pickImage(source: source);
    if (picked == null) return;

    setState(() {
      _selectedImage = File(picked.path);
      _result = null;
      _errorMessage = null;
    });

    await _scanImage(_selectedImage!, method: method);
  }

  Future<void> _scanImage(File imageFile, {String method = 'hybrid'}) async {
    setState(() {
      _isLoading = true;
      _errorMessage = null;
    });

    try {
      final bytes = await imageFile.readAsBytes();
      if (bytes.length < 1024) {
        setState(() {
          _errorMessage = 'The selected photo appears incomplete or empty '
              '(${bytes.length} bytes). Please try picking it again.';
          _isLoading = false;
        });
        return;
      }

      final path = switch (method) {
        'ai-text' => '/scan-product-ai',
        'ai-vision' => '/scan-product-ai-vision',
        'regex' => '/scan-product',
        _ => '/scan-product-hybrid',
      };
      final uri = Uri.parse('$baseUrl$path');
      final request = http.MultipartRequest('POST', uri);
      request.files.add(
        http.MultipartFile.fromBytes('file', bytes,
            filename: imageFile.path.split('/').last),
      );

      final streamedResponse = await request.send();
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        setState(() {
          _result = jsonDecode(response.body) as Map<String, dynamic>;
          _scanMethod = method;
          _isLoading = false;
        });
      } else {
        String detail;
        try {
          detail = (jsonDecode(response.body) as Map<String, dynamic>)['detail']
                  ?.toString() ??
              response.body;
        } catch (_) {
          detail = response.body;
        }
        setState(() {
          _errorMessage = 'Server error (${response.statusCode}): $detail';
          _isLoading = false;
        });
      }
    } catch (e) {
      setState(() {
        _errorMessage = 'Could not reach the backend at $baseUrl.\n'
            'Check that api.py (uvicorn) is running and baseUrl is correct.\n\nDetails: $e';
        _isLoading = false;
      });
    }
  }

  Color _statusColor(String status) {
    switch (status) {
      case 'PASS':
        return Colors.green;
      case 'FAIL':
        return Colors.red;
      case 'REVIEW':
        return Colors.orange;
      default:
        return Colors.grey;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Legal Metrology Scanner'),
        actions: [
          IconButton(
            icon: const Icon(Icons.history),
            tooltip: 'Scan History',
            onPressed: () {
              Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (context) => const HistoryPage(),
                ),
              );
            },
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            if (_selectedImage != null)
              ClipRRect(
                borderRadius: BorderRadius.circular(8),
                child:
                    Image.file(_selectedImage!, height: 220, fit: BoxFit.cover),
              ),
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    icon: const Icon(Icons.camera_alt),
                    label: const Text('Take Photo'),
                    onPressed: _isLoading
                        ? null
                        : () => _pickImage(ImageSource.camera),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: OutlinedButton.icon(
                    icon: const Icon(Icons.photo_library),
                    label: const Text('From Gallery'),
                    onPressed: _isLoading
                        ? null
                        : () => _pickImage(ImageSource.gallery),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Center(
              child: TextButton.icon(
                icon: const Icon(Icons.tune, size: 16),
                label: const Text('Other scan methods'),
                onPressed: _isLoading ? null : _showOtherScanOptions,
              ),
            ),
            if (_result != null)
              Center(
                child: Text(
                  'Scan method used: $_scanMethod',
                  style: TextStyle(fontSize: 11, color: Colors.grey.shade600),
                ),
              ),
            const SizedBox(height: 24),
            if (_isLoading)
              const Padding(
                padding: EdgeInsets.symmetric(vertical: 24),
                child: Center(child: CircularProgressIndicator()),
              ),
            if (_errorMessage != null)
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.red.shade50,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: Colors.red.shade200),
                ),
                child: Text(_errorMessage!,
                    style: TextStyle(color: Colors.red.shade800)),
              ),
            if (_result != null) _buildProductInfo(_result!),
            if (_result != null) _buildComplianceReport(_result!),
            if (_result != null) _buildNutritionSection(_result!),
          ],
        ),
      ),
    );
  }

  // ---- SECTION 1: basic product info (manufacturer, net qty, MRP, dates, etc.) ----
  Widget _buildProductInfo(Map<String, dynamic> result) {
    final product = result['product'] as Map<String, dynamic>? ?? {};

    const fields = {
      'manufacturer_name': 'Manufacturer',
      'manufacturer_address': 'Manufacturer Address',
      'net_quantity': 'Net Quantity',
      'mrp': 'MRP',
      'manufacturing_date': 'Manufacturing Date',
      'expiry_date': 'Expiry Date',
      'batch_number': 'Batch Number',
      'consumer_care_phone': 'Consumer Care Phone',
      'ingredients': 'Ingredients',
    };

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const Divider(height: 32),
        Text('Product Information',
            style: Theme.of(context).textTheme.titleLarge),
        const SizedBox(height: 8),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(12),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: fields.entries.map((entry) {
                final value = (product[entry.key] ?? '').toString();
                return Padding(
                  padding: const EdgeInsets.symmetric(vertical: 4),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      SizedBox(
                        width: 150,
                        child: Text(entry.value,
                            style:
                                const TextStyle(fontWeight: FontWeight.w500)),
                      ),
                      Expanded(
                        child: Text(
                          value.isEmpty ? 'Not detected' : value,
                          style: TextStyle(
                              color:
                                  value.isEmpty ? Colors.grey : Colors.black87),
                        ),
                      ),
                    ],
                  ),
                );
              }).toList(),
            ),
          ),
        ),
      ],
    );
  }

  // ---- SECTION 2: Legal Metrology compliance PASS/FAIL/REVIEW ----
  Widget _buildComplianceReport(Map<String, dynamic> result) {
    final report = result['report'] as Map<String, dynamic>? ?? {};
    final summary = report['summary'] as Map<String, dynamic>? ?? {};
    final results = (report['results'] as List<dynamic>?) ?? [];

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const Divider(height: 32),
        Text('Legal Metrology Compliance',
            style: Theme.of(context).textTheme.titleLarge),
        const SizedBox(height: 8),
        Wrap(
          spacing: 8,
          children: [
            Chip(
                label: Text('PASS: ${summary['passed'] ?? 0}'),
                backgroundColor: Colors.green.shade100),
            Chip(
                label: Text('FAIL: ${summary['failed'] ?? 0}'),
                backgroundColor: Colors.red.shade100),
            Chip(
                label: Text('REVIEW: ${summary['review'] ?? 0}'),
                backgroundColor: Colors.orange.shade100),
          ],
        ),
        const SizedBox(height: 16),
        if (results.isEmpty)
          const Text('No compliance checks were run.',
              style: TextStyle(fontStyle: FontStyle.italic)),
        ...results.map((r) {
          final item = r as Map<String, dynamic>;
          final status = (item['status'] ?? 'REVIEW').toString();
          return Card(
            margin: const EdgeInsets.only(bottom: 8),
            child: ListTile(
              leading: CircleAvatar(
                backgroundColor: _statusColor(status),
                child: Text(
                  status.isNotEmpty ? status[0] : '?',
                  style: const TextStyle(
                      color: Colors.white, fontWeight: FontWeight.bold),
                ),
              ),
              title:
                  Text('Rule ${item['rule_number']}: ${item['title'] ?? ''}'),
              subtitle: Text((item['evidence'] ?? '').toString()),
              trailing: Text(
                status,
                style: TextStyle(
                    color: _statusColor(status), fontWeight: FontWeight.bold),
              ),
            ),
          );
        }),
      ],
    );
  }

  // ---- SECTION 3: Nutrition facts + marketing claims ----
  Widget _buildNutritionSection(Map<String, dynamic> result) {
    final nutritionAnalysis =
        result['nutrition_analysis'] as Map<String, dynamic>?;
    if (nutritionAnalysis == null) return const SizedBox.shrink();

    final nutrition =
        nutritionAnalysis['nutrition'] as Map<String, dynamic>? ?? {};
    final claims = (nutritionAnalysis['claims'] as List<dynamic>?) ?? [];

    const fieldLabels = {
      'serving_size': 'Serving Size',
      'calories': 'Calories (kcal)',
      'protein_g': 'Protein (g)',
      'carbohydrates_g': 'Carbohydrates (g)',
      'sugar_g': 'Sugar (g)',
      'added_sugar_g': 'Added Sugar (g)',
      'fat_g': 'Fat (g)',
      'sodium_mg': 'Sodium (mg)',
    };

    final anyNutritionFound = nutrition.values
        .any((v) => v != null && v.toString().trim().isNotEmpty);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const Divider(height: 32),
        Text('Nutrition & Claims',
            style: Theme.of(context).textTheme.titleLarge),
        Text(
          'Separate from Legal Metrology compliance above — never affects PASS/FAIL.',
          style: Theme.of(context)
              .textTheme
              .bodySmall
              ?.copyWith(color: Colors.grey.shade600),
        ),
        const SizedBox(height: 12),
        if (!anyNutritionFound)
          const Text('No nutrition table detected on this label.',
              style: TextStyle(fontStyle: FontStyle.italic))
        else
          Card(
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: fieldLabels.entries.map((entry) {
                  final value = (nutrition[entry.key] ?? '').toString();
                  return Padding(
                    padding: const EdgeInsets.symmetric(vertical: 4),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text(entry.value,
                            style:
                                const TextStyle(fontWeight: FontWeight.w500)),
                        Text(value.isEmpty ? 'Not detected' : value,
                            style: TextStyle(
                                color: value.isEmpty
                                    ? Colors.grey
                                    : Colors.black)),
                      ],
                    ),
                  );
                }).toList(),
              ),
            ),
          ),
        const SizedBox(height: 16),
        Text('Marketing Claims Detected (${claims.length})',
            style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(height: 8),
        if (claims.isEmpty)
          const Text('No marketing claims detected on this label.',
              style: TextStyle(fontStyle: FontStyle.italic))
        else
          ...claims.map((c) {
            final claim = c as Map<String, dynamic>;
            final status = (claim['status'] ?? 'NEEDS_VERIFICATION').toString();
            final isSupported = status == 'SUPPORTED';
            return Card(
              margin: const EdgeInsets.only(bottom: 8),
              child: ListTile(
                leading: Icon(isSupported ? Icons.check_circle : Icons.help,
                    color: isSupported ? Colors.green : Colors.orange),
                title: Text((claim['claim_text'] ?? '').toString()),
                subtitle: Text((claim['evidence'] ?? '').toString()),
                trailing: Text(
                  status.replaceAll('_', ' '),
                  style: TextStyle(
                    color: isSupported ? Colors.green : Colors.orange,
                    fontWeight: FontWeight.bold,
                    fontSize: 11,
                  ),
                ),
              ),
            );
          }),
        const SizedBox(height: 16),
        ElevatedButton.icon(
          icon: const Icon(Icons.save),
          label: const Text('Save to History'),
          onPressed: _saveToHistory,
        ),
      ],
    );
  }
}
