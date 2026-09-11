// ADMIN SIDE UI: upload the official Legal Metrology PDF and show a
// summary of what got extracted (pages processed, rules found/missing,
// compliance checks built). This mirrors the JSON returned by
// POST /admin/upload-rules-pdf in api.py — no new logic here, just a
// display of what the backend already computed.

import 'dart:convert';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

import 'config.dart';

class AdminUploadPage extends StatefulWidget {
  const AdminUploadPage({super.key});

  @override
  State<AdminUploadPage> createState() => _AdminUploadPageState();
}

class _AdminUploadPageState extends State<AdminUploadPage> {
  static const String baseUrl = AppConfig.baseUrl;

  String? _pickedFileName;
  bool _isLoading = false;
  String? _errorMessage;
  Map<String, dynamic>? _summary;

  Future<void> _pickAndUploadPdf() async {
    final FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['pdf'],
      withData: true, // load bytes directly, avoids any path/file-timing issues
    );

    if (result == null || result.files.isEmpty) return;

    final pickedFile = result.files.single;
    final bytes = pickedFile.bytes;

    if (bytes == null || bytes.length < 1024) {
      setState(() {
        _errorMessage =
            'The selected PDF appears incomplete or empty. Please try again.';
        _summary = null;
      });
      return;
    }

    setState(() {
      _pickedFileName = pickedFile.name;
      _isLoading = true;
      _errorMessage = null;
      _summary = null;
    });

    try {
      final uri = Uri.parse('$baseUrl/admin/upload-rules-pdf');
      final request = http.MultipartRequest('POST', uri);
      request.files.add(
        http.MultipartFile.fromBytes('file', bytes, filename: pickedFile.name),
      );

      final streamedResponse = await request.send();
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        setState(() {
          _summary = jsonDecode(response.body) as Map<String, dynamic>;
          _isLoading = false;
        });
      } else {
        setState(() {
          _errorMessage =
              'Server error (${response.statusCode}): ${response.body}';
          _isLoading = false;
        });
      }
    } catch (e) {
      setState(() {
        _errorMessage = 'Could not reach the backend at $baseUrl.\n'
            'Check that api.py is running and the URL in config.dart is correct.\n\nDetails: $e';
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Admin: Upload Rules PDF')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text(
              'Upload the official Legal Metrology (Packaged Commodities) '
              'Rules PDF. The backend will convert every page to an image, '
              'run OCR, extract individual rules, and rebuild the '
              'compliance checklist used by the scanner.',
              style: Theme.of(context).textTheme.bodyMedium,
            ),
            const SizedBox(height: 20),
            ElevatedButton.icon(
              icon: const Icon(Icons.upload_file),
              label: const Text('Pick & Upload PDF'),
              onPressed: _isLoading ? null : _pickAndUploadPdf,
            ),
            if (_pickedFileName != null) ...[
              const SizedBox(height: 8),
              Text('Selected: $_pickedFileName',
                  style: Theme.of(context).textTheme.bodySmall),
            ],
            const SizedBox(height: 24),
            if (_isLoading) const Center(child: CircularProgressIndicator()),
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
            if (_summary != null) _buildSummaryView(_summary!),
          ],
        ),
      ),
    );
  }

  Widget _buildSummaryView(Map<String, dynamic> summary) {
    final foundNumbers = (summary['rule_numbers_found'] as List<dynamic>);
    final missingNumbers = (summary['rule_numbers_missing'] as List<dynamic>);
    final duplicateNumbers =
        (summary['duplicate_rule_numbers'] as List<dynamic>);
    final missingCompliance =
        (summary['compliance_rule_numbers_missing'] as List<dynamic>);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const Divider(height: 32),
        Text('Extraction Summary',
            style: Theme.of(context).textTheme.titleLarge),
        const SizedBox(height: 12),
        _summaryRow('Pages processed', '${summary['pages_processed']}'),
        _summaryRow('Rules extracted', '${summary['rules_extracted']}'),
        _summaryRow(
            'Compliance checks built', '${summary['compliance_checks_built']}'),
        const SizedBox(height: 16),
        Text('Rule numbers found (${foundNumbers.length}):',
            style: Theme.of(context).textTheme.titleSmall),
        Text(foundNumbers.join(', '),
            style: Theme.of(context).textTheme.bodyMedium),
        const SizedBox(height: 12),
        if (missingNumbers.isNotEmpty) ...[
          Text(
            'Rule numbers NOT found (${missingNumbers.length}) — needs manual review:',
            style: Theme.of(context)
                .textTheme
                .titleSmall
                ?.copyWith(color: Colors.orange.shade800),
          ),
          Text(missingNumbers.join(', '),
              style: Theme.of(context).textTheme.bodyMedium),
          const SizedBox(height: 12),
        ],
        if (duplicateNumbers.isNotEmpty) ...[
          Text(
            'Duplicate rule numbers detected (check OCR quality):',
            style: Theme.of(context)
                .textTheme
                .titleSmall
                ?.copyWith(color: Colors.red.shade800),
          ),
          Text(duplicateNumbers.join(', '),
              style: Theme.of(context).textTheme.bodyMedium),
          const SizedBox(height: 12),
        ],
        if (missingCompliance.isNotEmpty) ...[
          Text(
            'Curated compliance checks skipped (source rule not found):',
            style: Theme.of(context)
                .textTheme
                .titleSmall
                ?.copyWith(color: Colors.orange.shade800),
          ),
          Text(missingCompliance.join(', '),
              style: Theme.of(context).textTheme.bodyMedium),
        ],
      ],
    );
  }

  Widget _summaryRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(fontWeight: FontWeight.w500)),
          Text(value),
        ],
      ),
    );
  }
}
