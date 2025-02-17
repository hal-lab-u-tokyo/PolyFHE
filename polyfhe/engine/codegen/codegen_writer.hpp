#pragma once

#include <fstream>
#include <sstream>
#include <string>

#include "polyfhe/core/logger.hpp"

namespace polyfhe {
namespace engine {

class CodeWriter {
public:
    CodeWriter() : indent(0), m_pending_indent(true) {}
    void operator+=(const std::string& str) { *this << str; }

    size_t indent;

    template <typename T>
    friend CodeWriter& operator<<(CodeWriter& out, const T& obj) {
        std::stringstream ss;
        ss << obj;

        for (char c : ss.str()) {
            if (c == '\n') {
                out.m_pending_indent = true;
            } else {
                if (out.m_pending_indent) {
                    out.m_pending_indent = false;
                    for (size_t i = 0; i < out.indent; i++) {
                        out.m_ss << "    ";
                    }
                }
            }
            out.m_ss << c;
        }

        return out;
    }

    void block_begin() {
        *this << "{\n";
        indent++;
    }

    void block_end() {
        indent--;
        *this << "}\n";
    }

    void indent_inc() { indent++; }

    void indent_dec() { indent--; }

    void write_to_file(const std::string& filename, const bool append = false) {
        auto flag = append ? std::ios_base::out | std::ios_base::app
                           : std::ios_base::out;
        std::ofstream out(filename, flag);
        if (!out) {
            LOG_ERROR("Cannot open file: %s\n", filename.c_str());
        }
        out << m_ss.str();
    }

private:
    std::string symbol;
    std::string write_to;
    std::stringstream m_ss;
    bool m_pending_indent;
};

} // namespace engine
} // namespace polyfhe