// History screen: search/list saved scans via GET /scans, view one
// via GET /scans/{id}. New file — does not touch your existing
// scanner or admin screens.

import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'config.dart';

class HistoryPage extends StatefulWidget {
  const HistoryPage({super.key});
  @override
  State<HistoryPage> createState() => _HistoryPageState();
}

class _HistoryPageState extends State<HistoryPage> {
  static const String baseUrl = AppConfig.baseUrl;
  final _searchController = TextEditingController();
  String? _statusFilter;
  bool _isLoading = false;
  String? _error;
  List<dynamic> _results = [];

  @override
  void initState() {
    super.initState();
    _search();
  }

  Color _statusColor(String? s) {
    switch (s) {
      case 'COMPLIANT':
        return Colors.green;
      case 'NON_COMPLIANT':
        return Colors.red;
      case 'NEEDS_REVIEW':
        return Colors.orange;
      default:
        return Colors.grey;
    }
  }

  Future<void> _search() async {
    setState(() {
      _isLoading = true;
      _error = null;
    });
    try {
      final params = <String, String>{};
      if (_searchController.text.trim().isNotEmpty) {
        params['q'] = _searchController.text.trim();
      }
      if (_statusFilter != null) params['status'] = _statusFilter!;
      final uri = Uri.parse('$baseUrl/scans').replace(queryParameters: params);
      final res = await http.get(uri);
      if (res.statusCode == 200) {
        final body = jsonDecode(res.body) as Map<String, dynamic>;
        setState(() {
          _results = body['results'] as List<dynamic>;
          _isLoading = false;
        });
      } else {
        setState(() {
          _error = 'Server error (${res.statusCode})';
          _isLoading = false;
        });
      }
    } catch (e) {
      setState(() {
        _error = 'Could not reach backend: $e';
        _isLoading = false;
      });
    }
  }

  Future<void> _openDetail(String id) async {
    final res = await http.get(Uri.parse('$baseUrl/scans/$id'));
    if (res.statusCode != 200 || !mounted) return;
    final scan = jsonDecode(res.body) as Map<String, dynamic>;
    final product = scan['product'] as Map<String, dynamic>? ?? {};
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      builder: (context) => DraggableScrollableSheet(
        expand: false,
        initialChildSize: 0.7,
        builder: (context, scrollController) => ListView(
          controller: scrollController,
          padding: const EdgeInsets.all(16),
          children: [
            Text(product['manufacturer_name'] ?? 'Unknown manufacturer',
                style: Theme.of(context).textTheme.titleLarge),
            Text('Status: ${scan['overall_status']}',
                style: TextStyle(color: _statusColor(scan['overall_status']))),
            const Divider(),
            ...product.entries.map((e) => Padding(
                  padding: const EdgeInsets.symmetric(vertical: 2),
                  child: Text('${e.key}: ${e.value}'),
                )),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Scan History')),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(12),
            child: Row(children: [
              Expanded(
                child: TextField(
                  controller: _searchController,
                  decoration: const InputDecoration(
                      hintText: 'Search manufacturer / batch / MRP',
                      prefixIcon: Icon(Icons.search)),
                  onSubmitted: (_) => _search(),
                ),
              ),
              const SizedBox(width: 8),
              DropdownButton<String?>(
                value: _statusFilter,
                hint: const Text('Status'),
                items: const [
                  DropdownMenuItem(value: null, child: Text('All')),
                  DropdownMenuItem(
                      value: 'COMPLIANT', child: Text('Compliant')),
                  DropdownMenuItem(
                      value: 'NON_COMPLIANT', child: Text('Non-compliant')),
                  DropdownMenuItem(
                      value: 'NEEDS_REVIEW', child: Text('Needs review')),
                ],
                onChanged: (v) {
                  setState(() => _statusFilter = v);
                  _search();
                },
              ),
            ]),
          ),
          if (_isLoading) const LinearProgressIndicator(),
          if (_error != null)
            Padding(
              padding: const EdgeInsets.all(12),
              child: Text(_error!, style: const TextStyle(color: Colors.red)),
            ),
          Expanded(
            child: ListView.builder(
              itemCount: _results.length,
              itemBuilder: (context, i) {
                final r = _results[i] as Map<String, dynamic>;
                return ListTile(
                  leading: CircleAvatar(
                    backgroundColor: _statusColor(r['overall_status']),
                    child: Text(r['overall_status']?[0] ?? '?',
                        style: const TextStyle(color: Colors.white)),
                  ),
                  title: Text(
                      r['manufacturer_name']?.toString().isNotEmpty == true
                          ? r['manufacturer_name']
                          : 'Unknown manufacturer'),
                  subtitle: Text(
                      'Batch ${r['batch_number'] ?? '-'} · ${r['created_at']}'),
                  onTap: () => _openDetail(r['id']),
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}
