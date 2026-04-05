#pragma once

#include <functional>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>

// ======================================================
// ArgParser — generic command-line argument parser
//
// Usage:
//   Derive a struct from ArgParser.
//   In the constructor, call add_double / add_int /
//   add_string / add_bool for each parameter.
//   Call parse(argc, argv) in main().
//
// Supported syntax:  --key value   (for all types)
//                    --flag        (for bool, sets true)
// ======================================================

struct ArgParser {

    struct Spec {
        std::string key;
        std::string description;
        std::string default_str;
        std::function<bool(const std::string&)> setter; // returns false on bad input
        std::function<std::string()> getter;
    };

    std::vector<Spec> specs;

    // --- registration helpers ---

    void add_double(const std::string& key, double& field,
                    double def, const std::string& desc) {
        field = def;
        specs.push_back({key, desc, std::to_string(def),
            [&field](const std::string& v) {
                try { field = std::stod(v); return true; }
                catch (...) { return false; }
            },
            [&field]() { return std::to_string(field); }});
    }

    void add_int(const std::string& key, int& field,
                 int def, const std::string& desc) {
        field = def;
        specs.push_back({key, desc, std::to_string(def),
            [&field](const std::string& v) {
                try { field = std::stoi(v); return true; }
                catch (...) { return false; }
            },
            [&field]() { return std::to_string(field); }});
    }

    void add_string(const std::string& key, std::string& field,
                    const std::string& def, const std::string& desc) {
        field = def;
        specs.push_back({key, desc, def,
            [&field](const std::string& v) {
                field = v; return true;
            },
            [&field]() { return field; }});
    }

    // bool: --flag sets true, --flag false/0 sets false
    void add_bool(const std::string& key, bool& field,
                  bool def, const std::string& desc) {
        field = def;
        specs.push_back({key, desc, def ? "true" : "false",
            [&field](const std::string& v) {
                if (v == "true"  || v == "1") { field = true;  return true; }
                if (v == "false" || v == "0") { field = false; return true; }
                return false;
            },
            [&field]() { return field ? std::string("true") : std::string("false"); }});
    }

    // --- parsing ---

    // Returns true on success; prints errors and returns false on bad input.
    // Unrecognised keys are treated as errors.
    bool parse(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            std::string tok(argv[i]);

            if (tok == "--help" || tok == "-h") {
                print_usage(argv[0]);
                return false;
            }

            if (tok.rfind("--", 0) != 0) {
                std::cerr << "Error: unexpected token '" << tok << "'\n";
                print_usage(argv[0]);
                return false;
            }

            std::string key = tok.substr(2);
            Spec* spec = find_spec(key);

            if (!spec) {
                std::cerr << "Error: unknown argument '--" << key << "'\n";
                print_usage(argv[0]);
                return false;
            }

            // bool flag with no following value
            if (spec->default_str == "true" || spec->default_str == "false") {
                if (i + 1 >= argc || std::string(argv[i+1]).rfind("--", 0) == 0) {
                    spec->setter("true");
                    continue;
                }
            }

            if (i + 1 >= argc) {
                std::cerr << "Error: '--" << key << "' requires a value\n";
                return false;
            }

            std::string val(argv[++i]);
            if (!spec->setter(val)) {
                std::cerr << "Error: invalid value '" << val
                          << "' for '--" << key << "'\n";
                return false;
            }
        }
        return true;
    }

    // Serialize all registered fields as "key=value\n" lines.
    void serialize(const std::string& path) const {
        std::ofstream out(path);
        if (!out) { std::cerr << "Error: cannot write args file: " << path << "\n"; return; }
        for (const auto& s : specs)
            out << s.key << "=" << s.getter() << "\n";
    }

    // Deserialize by reading "key=value" lines and invoking each field's setter.
    bool deserialize(const std::string& path) {
        std::ifstream in(path);
        if (!in) { std::cerr << "Error: cannot read args file: " << path << "\n"; return false; }
        std::string line;
        while (std::getline(in, line)) {
            auto eq = line.find('=');
            if (eq == std::string::npos) continue;
            std::string key = line.substr(0, eq);
            std::string val = line.substr(eq + 1);
            Spec* spec = find_spec(key);
            if (spec) spec->setter(val);
        }
        return in.eof();
    }

    void print_usage(const std::string& program_name) const {
        std::cout << "Usage: " << program_name << " [options]\n\nOptions:\n";
        for (const auto& s : specs) {
            std::cout << "  --" << s.key
                      << "\n      " << s.description
                      << "  (default: " << s.default_str << ")\n";
        }
        std::cout << "  --help\n      Print this message\n";
    }

private:
    Spec* find_spec(const std::string& key) {
        for (auto& s : specs)
            if (s.key == key) return &s;
        return nullptr;
    }
};
