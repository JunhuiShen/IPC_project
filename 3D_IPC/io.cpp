#include "io.h"

#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct DataLine {
    std::string text;
    std::size_t number = 0;
};

[[noreturn]] void throw_format_error(
    const std::string& filename,
    const std::size_t line_number,
    const std::string& message)
{
    std::ostringstream out;
    out << "Invalid TetGen file '" << filename << "'";
    if (line_number != 0)
        out << " at line " << line_number;
    out << ": " << message;
    throw std::runtime_error(out.str());
}

bool next_data_line(
    std::ifstream& input,
    std::size_t& line_number,
    DataLine& result)
{
    std::string line;
    while (std::getline(input, line)) {
        ++line_number;

        const std::size_t comment = line.find('#');
        if (comment != std::string::npos)
            line.erase(comment);

        std::istringstream tokens(line);
        std::string first_token;
        if (!(tokens >> first_token))
            continue;

        result = {std::move(line), line_number};
        return true;
    }

    if (input.bad())
        throw std::runtime_error("I/O error while reading a TetGen file");
    return false;
}

DataLine require_data_line(
    std::ifstream& input,
    std::size_t& line_number,
    const std::string& filename,
    const std::string& description)
{
    DataLine result;
    if (!next_data_line(input, line_number, result))
        throw_format_error(filename, line_number, "missing " + description);
    return result;
}

template <typename T>
void read_required_value(
    std::istringstream& input,
    T& value,
    const std::string& filename,
    const std::size_t line_number,
    const std::string& description)
{
    if (!(input >> value))
        throw_format_error(filename, line_number, "missing or invalid " + description);
}

void require_end_of_record(
    std::istringstream& input,
    const std::string& filename,
    const std::size_t line_number)
{
    std::string extra;
    if (input >> extra)
        throw_format_error(filename, line_number, "unexpected extra record field");
}

std::ifstream open_tetgen_file(const std::string& filename)
{
    std::ifstream input(filename);
    if (!input)
        throw std::runtime_error("Unable to open TetGen file '" + filename + "'");
    return input;
}

std::size_t checked_count(
    const long long count,
    const std::size_t entries_per_record,
    const std::size_t maximum_entries,
    const std::string& filename,
    const std::size_t line_number,
    const std::string& description)
{
    if (count < 0)
        throw_format_error(filename, line_number, description + " must be nonnegative");

    const auto unsigned_count = static_cast<unsigned long long>(count);
    if (unsigned_count > std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("TetGen " + description + " is too large");

    const std::size_t result = static_cast<std::size_t>(unsigned_count);
    if (entries_per_record != 0
        && result > maximum_entries / entries_per_record)
        throw std::overflow_error("TetGen " + description + " is too large");
    return result;
}

std::size_t checked_auxiliary_count(
    const long long count,
    const std::string& filename,
    const std::size_t line_number,
    const std::string& description)
{
    if (count < 0)
        throw_format_error(filename, line_number, description + " must be nonnegative");
    const auto unsigned_count = static_cast<unsigned long long>(count);
    if (unsigned_count > std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("TetGen " + description + " is too large");
    return static_cast<std::size_t>(unsigned_count);
}

long long expected_record_id(
    const std::size_t record,
    const int input_index_offset)
{
    const auto max_id = static_cast<unsigned long long>(
        std::numeric_limits<long long>::max() - input_index_offset);
    if (record > max_id)
        throw std::overflow_error("TetGen record ID is too large");
    return static_cast<long long>(record) + input_index_offset;
}

void require_sequential_record_id(
    const long long actual,
    const std::size_t record,
    const int input_index_offset,
    const std::string& filename,
    const std::size_t line_number)
{
    if (actual != expected_record_id(record, input_index_offset))
        throw_format_error(filename, line_number, "record IDs must be consecutive");
}

int normalize_connectivity_index(
    const long long input_index,
    const int input_index_offset,
    const std::string& filename,
    const std::size_t line_number)
{
    if (input_index < input_index_offset)
        throw_format_error(filename, line_number, "connectivity index is below the input index base");

    const long long normalized = input_index - input_index_offset;
    if (normalized > std::numeric_limits<int>::max())
        throw std::overflow_error("TetGen connectivity index is too large");
    return static_cast<int>(normalized);
}

void require_no_additional_records(
    std::ifstream& input,
    std::size_t& line_number,
    const std::string& filename)
{
    DataLine extra;
    if (next_data_line(input, line_number, extra))
        throw_format_error(filename, extra.number, "more records than declared in the header");
}

void validate_marker_count(
    const long long marker_count,
    const std::string& filename,
    const std::size_t line_number)
{
    if (marker_count != 0 && marker_count != 1)
        throw_format_error(filename, line_number, "boundary-marker count must be zero or one");
}

void validate_tet_mesh_output(
    const std::vector<Vec3>& positions,
    const std::vector<int>& tets,
    const char* function_name)
{
    if (positions.empty())
        throw std::invalid_argument(std::string(function_name) + ": positions cannot be empty");
    if (tets.size() % 4 != 0) {
        throw std::invalid_argument(
            std::string(function_name)
            + ": tets must contain four indices per element");
    }
    if (positions.size()
            > static_cast<std::size_t>(std::numeric_limits<int>::max())
        || tets.size() / 4
            > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            std::string(function_name) + ": mesh is too large");
    }
    for (const Vec3& position : positions) {
        if (!position.allFinite()) {
            throw std::invalid_argument(
                std::string(function_name) + ": positions must be finite");
        }
    }
    for (const int node : tets) {
        if (node < 0 || node >= static_cast<int>(positions.size())) {
            throw std::out_of_range(
                std::string(function_name)
                + ": tetrahedron node index is out of range");
        }
    }
}

} // namespace

void read_tetgen_nodes(
    const std::string& filename,
    std::vector<Vec3>& positions,
    const bool zero_based_index)
{
    std::ifstream input = open_tetgen_file(filename);
    std::size_t line_number = 0;

    const DataLine header = require_data_line(
        input, line_number, filename, "node header");
    std::istringstream header_stream(header.text);
    long long node_count_value = 0;
    long long dimension = 0;
    long long attribute_count_value = 0;
    long long marker_count = 0;
    read_required_value(header_stream, node_count_value, filename, header.number, "node count");
    read_required_value(header_stream, dimension, filename, header.number, "dimension");
    read_required_value(header_stream, attribute_count_value, filename, header.number, "attribute count");
    read_required_value(header_stream, marker_count, filename, header.number, "boundary-marker count");
    require_end_of_record(header_stream, filename, header.number);

    if (dimension != 3)
        throw_format_error(filename, header.number, "only three-dimensional node files are supported");
    validate_marker_count(marker_count, filename, header.number);

    const std::size_t node_count = checked_count(
        node_count_value, 1, std::vector<Vec3>().max_size(), filename,
        header.number, "node count");
    const std::size_t attribute_count = checked_auxiliary_count(
        attribute_count_value, filename, header.number, "attribute count");
    const int input_index_offset = zero_based_index ? 0 : 1;

    std::vector<Vec3> parsed(node_count, Vec3::Zero());
    for (std::size_t i = 0; i < node_count; ++i) {
        const DataLine record = require_data_line(
            input, line_number, filename, "node record");
        std::istringstream record_stream(record.text);

        long long record_id = 0;
        read_required_value(record_stream, record_id, filename, record.number, "node record ID");
        require_sequential_record_id(
            record_id, i, input_index_offset, filename, record.number);

        for (int axis = 0; axis < 3; ++axis) {
            double coordinate = 0.0;
            read_required_value(
                record_stream, coordinate, filename, record.number,
                "node coordinate");
            if (!std::isfinite(coordinate))
                throw_format_error(filename, record.number, "node coordinate must be finite");
            parsed[i][axis] = coordinate;
        }

        for (std::size_t attribute = 0; attribute < attribute_count; ++attribute) {
            double ignored = 0.0;
            read_required_value(
                record_stream, ignored, filename, record.number,
                "node attribute");
        }
        if (marker_count == 1) {
            long long ignored_marker = 0;
            read_required_value(
                record_stream, ignored_marker, filename, record.number,
                "node boundary marker");
        }
        require_end_of_record(record_stream, filename, record.number);
    }

    require_no_additional_records(input, line_number, filename);
    positions.swap(parsed);
}

void read_tetgen_tets(
    const std::string& filename,
    std::vector<int>& tets,
    const bool zero_based_index)
{
    std::ifstream input = open_tetgen_file(filename);
    std::size_t line_number = 0;

    const DataLine header = require_data_line(
        input, line_number, filename, "element header");
    std::istringstream header_stream(header.text);
    long long tet_count_value = 0;
    long long nodes_per_tet = 0;
    long long attribute_count_value = 0;
    read_required_value(header_stream, tet_count_value, filename, header.number, "tetrahedron count");
    read_required_value(header_stream, nodes_per_tet, filename, header.number, "nodes per tetrahedron");
    read_required_value(header_stream, attribute_count_value, filename, header.number, "attribute count");
    require_end_of_record(header_stream, filename, header.number);

    if (nodes_per_tet != 4)
        throw_format_error(filename, header.number, "only four-node tetrahedra are supported");

    const std::size_t tet_count = checked_count(
        tet_count_value, 4, std::vector<int>().max_size(), filename,
        header.number, "tetrahedron count");
    const std::size_t attribute_count = checked_auxiliary_count(
        attribute_count_value, filename, header.number, "attribute count");
    const int input_index_offset = zero_based_index ? 0 : 1;

    std::vector<int> parsed(4 * tet_count);
    for (std::size_t i = 0; i < tet_count; ++i) {
        const DataLine record = require_data_line(
            input, line_number, filename, "tetrahedron record");
        std::istringstream record_stream(record.text);

        long long record_id = 0;
        read_required_value(record_stream, record_id, filename, record.number, "tetrahedron record ID");
        require_sequential_record_id(
            record_id, i, input_index_offset, filename, record.number);

        for (std::size_t local = 0; local < 4; ++local) {
            long long node_id = 0;
            read_required_value(
                record_stream, node_id, filename, record.number,
                "tetrahedron node index");
            parsed[4 * i + local] = normalize_connectivity_index(
                node_id, input_index_offset, filename, record.number);
        }
        for (std::size_t attribute = 0; attribute < attribute_count; ++attribute) {
            double ignored = 0.0;
            read_required_value(
                record_stream, ignored, filename, record.number,
                "tetrahedron attribute");
        }
        require_end_of_record(record_stream, filename, record.number);
    }

    require_no_additional_records(input, line_number, filename);
    tets.swap(parsed);
}

void read_tetgen_faces(
    const std::string& filename,
    std::vector<int>& faces,
    const bool zero_based_index)
{
    std::ifstream input = open_tetgen_file(filename);
    std::size_t line_number = 0;

    const DataLine header = require_data_line(
        input, line_number, filename, "face header");
    std::istringstream header_stream(header.text);
    long long face_count_value = 0;
    long long marker_count = 0;
    read_required_value(header_stream, face_count_value, filename, header.number, "face count");
    read_required_value(header_stream, marker_count, filename, header.number, "boundary-marker count");
    require_end_of_record(header_stream, filename, header.number);
    validate_marker_count(marker_count, filename, header.number);

    const std::size_t face_count = checked_count(
        face_count_value, 3, std::vector<int>().max_size(), filename,
        header.number, "face count");
    const int input_index_offset = zero_based_index ? 0 : 1;

    std::vector<int> parsed(3 * face_count);
    for (std::size_t i = 0; i < face_count; ++i) {
        const DataLine record = require_data_line(
            input, line_number, filename, "face record");
        std::istringstream record_stream(record.text);

        long long record_id = 0;
        read_required_value(record_stream, record_id, filename, record.number, "face record ID");
        require_sequential_record_id(
            record_id, i, input_index_offset, filename, record.number);

        for (std::size_t local = 0; local < 3; ++local) {
            long long node_id = 0;
            read_required_value(
                record_stream, node_id, filename, record.number,
                "face node index");
            parsed[3 * i + local] = normalize_connectivity_index(
                node_id, input_index_offset, filename, record.number);
        }
        if (marker_count == 1) {
            long long ignored_marker = 0;
            read_required_value(
                record_stream, ignored_marker, filename, record.number,
                "face boundary marker");
        }
        require_end_of_record(record_stream, filename, record.number);
    }

    require_no_additional_records(input, line_number, filename);
    faces.swap(parsed);
}

void write_tet_mesh_obj(
    const std::string& filename,
    const std::vector<Vec3>& positions,
    const std::vector<int>& tets)
{
    validate_tet_mesh_output(positions, tets, "write_tet_mesh_obj");

    std::ofstream output(filename);
    if (!output)
        throw std::runtime_error("write_tet_mesh_obj: cannot open '" + filename + "'");

    output << std::setprecision(17);
    for (const Vec3& position : positions) {
        output << "v " << position.x() << " " << position.y() << " "
               << position.z() << '\n';
    }

    // Outward faces of a positively oriented tet, matching TetFace in
    // mesh.cpp/TGSL: {1,2,3}, {0,3,2}, {0,1,3}, {0,2,1}.
    constexpr int local_faces[4][3] = {
        {1, 2, 3},
        {0, 3, 2},
        {0, 1, 3},
        {0, 2, 1},
    };
    const std::size_t num_tets = tets.size() / 4;
    for (std::size_t element = 0; element < num_tets; ++element) {
        output << "g tet_" << element << '\n';
        for (const auto& face : local_faces) {
            output << "f";
            for (const int local : face)
                output << " " << tets[4 * element + local] + 1;
            output << '\n';
        }
    }

    output.close();
    if (!output)
        throw std::runtime_error("write_tet_mesh_obj: failed while writing '" + filename + "'");
}
