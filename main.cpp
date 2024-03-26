#include <iostream>
#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <cstdint>
#include <cstring>
#include <string>
#include <array>
#include <vector>
#define _USE_MATH_DEFINES
#include <cmath>
#include <limits>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <windows.h>

using namespace cv;
using namespace std;


const double NaN = std::numeric_limits<double>::quiet_NaN();

std::string deleteLastNCharacters(const std::string& str, const int n) {
    if (str.length() >= n) {
        return str.substr(0, str.length() - n);
    }
    else {
        // Handle cases where the string has less than n characters
        return "";
    }
}

bool isASCII(const std::string& filePath, std::string& firstLine) {
    std::ifstream file(filePath);

    std::getline(file, firstLine);

    // Check if the first line contains the "solid" keyword
    bool isAsciiFormat = (firstLine.find("solid") != std::string::npos);

    // Check if the file exists
    if (!file) {
        std::cerr << "Error: File does not exist: " << filePath << std::endl;
        return true;
    }

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filePath << std::endl;
        return true;
    }
    file.close();
    return isAsciiFormat;
}

void readSTL(const std::string& file, float angleomit,
    std::vector<std::vector<double>>& points,
    std::vector<std::vector<double>>& Nvecs,
    std::vector<std::vector<double>>& nvecpoints,
    float ZOFFSET) {

    // Open the file
    std::ifstream fileStream(file, std::ios::binary | std::ios::ate);

    // Get the file size
    std::streamsize fileSize = fileStream.tellg();
    fileStream.seekg(0, std::ios::beg);

    // Read the entire file into a vector
    std::vector<uint8_t> M(fileSize);
    fileStream.read(reinterpret_cast<char*>(M.data()), fileSize);
    fileStream.close();
    // Extract relevant information
    std::vector<uint8_t> info(M.begin() + 84, M.end());
    uint32_t nFaces = *reinterpret_cast<uint32_t*>(&M[80]);

    // Initialize vectors
    std::vector<std::vector<float>> nvecs(nFaces, std::vector<float>(6, 0.0f));
    std::vector<std::vector<float>> verts(3 * nFaces, std::vector<float>(3, 0.0f));

    for (uint32_t i = 0; i < nFaces; ++i) {
        std::vector<uint8_t> facet(info.begin() + 50 * i, info.begin() + 50 * (i + 1));

        std::vector<float> v1(3, 0.0f);
        std::vector<float> v2(3, 0.0f);
        std::vector<float> v3(3, 0.0f);
        if (facet.size() >= 48) {
            // v1
            std::memcpy(&v1[0], &facet[12], sizeof(float));
            std::memcpy(&v1[1], &facet[16], sizeof(float));
            std::memcpy(&v1[2], &facet[20], sizeof(float));

            // v2
            std::memcpy(&v2[0], &facet[24], sizeof(float));
            std::memcpy(&v2[1], &facet[28], sizeof(float));
            std::memcpy(&v2[2], &facet[32], sizeof(float));

            // v3
            std::memcpy(&v3[0], &facet[36], sizeof(float));
            std::memcpy(&v3[1], &facet[40], sizeof(float));
            std::memcpy(&v3[2], &facet[44], sizeof(float));
        }
        verts[3 * i] = { v1[0], v1[1], v1[2] };
        verts[3 * i + 1] = { v2[0], v2[1], v2[2] };
        verts[3 * i + 2] = { v3[0], v3[1], v3[2] };

        nvecs[i][0] = *reinterpret_cast<float*>(&facet[0]);
        nvecs[i][1] = *reinterpret_cast<float*>(&facet[4]);
        nvecs[i][2] = *reinterpret_cast<float*>(&facet[8]);
        nvecs[i][3] = (v1[0] + v2[0] + v3[0]) / 3;
        nvecs[i][4] = (v1[1] + v2[1] + v3[1]) / 3;
        nvecs[i][5] = (v1[2] + v2[2] + v3[2]) / 3;
    }
    // Filter nvecs
    std::vector<std::vector<float>> filteredNvecs;
    for (const auto& nvec : nvecs) {
        if (nvec[1] > 0 &&
            atan(nvec[1] / sqrt(nvec[0] * nvec[0] + nvec[2] * nvec[2])) > angleomit * M_PI / 180) {
            filteredNvecs.push_back(nvec);
        }
    }

    for (const auto& row : filteredNvecs) {
        std::vector<double> extractedColumns = { row[5], row[3], (row[4] + ZOFFSET) };
        std::vector<double> extractedColumns2 = { row[2], row[0], row[1] };
        nvecpoints.push_back(extractedColumns);
        Nvecs.push_back(extractedColumns2);
        points.push_back(extractedColumns);
    }
    for (const auto& row1 : verts) {
        std::vector<double> extractedColumns3 = { row1[2], row1[0], (row1[1] + ZOFFSET) };
        points.push_back(extractedColumns3);
    }
}

void negateFirstColumn(std::vector<std::vector<double>>& matrix) {
    for (auto& row : matrix) {
        row[0] = -row[0];
    }
}

double this_is_america(double celsius) {
    return (celsius * 9 / 5) + 32;
}

double goodcolumn(const std::vector<std::vector<double>>& matrix, size_t column) {
    if (matrix.empty() || matrix[0].size() <= column) {
        std::cerr << "Error: Invalid matrix or column index.\n";
        return 0.0; // Return a default value or handle the error as needed
    }
    else {
        return 1.0;
    }
}

double calculateMean(const std::vector<std::vector<double>>& matrix, size_t column) {
    goodcolumn(matrix, column);
    double sum = 0.0;
    for (const auto& row : matrix) {
        sum += row[column];
    }

    return sum / matrix.size();
}
double calculateMax(const std::vector<std::vector<double>>& matrix, size_t column) {
    goodcolumn(matrix, column);
    double maxVal = matrix[0][column];
    for (const auto& row : matrix) {
        maxVal = std::max(maxVal, row[column]);
    }

    return maxVal;
}
double calculateMin(const std::vector<std::vector<double>>& matrix, size_t column) {
    goodcolumn(matrix, column);
    double minVal = matrix[0][column];
    for (const auto& row : matrix) {
        minVal = std::min(minVal, row[column]);
    }

    return minVal;
}

std::vector<std::vector<double>> concatenateMatrices(const std::vector<std::vector<double>>& matrix1,
    const std::vector<std::vector<double>>& matrix2) {
    // Check if the matrices have the same number of rows
    if (matrix1.size() != matrix2.size()) {
        std::cerr << "Matrices have different numbers of rows and cannot be concatenated." << std::endl;
        return std::vector<std::vector<double>>();  // Return an empty matrix
    }
    // Concatenate matrices horizontally
    std::vector<std::vector<double>> concatenatedMatrix;
    for (size_t i = 0; i < matrix1.size(); ++i) {
        // Combine the rows of both matrices
        std::vector<double> combinedRow;
        combinedRow.insert(combinedRow.end(), matrix1[i].begin(), matrix1[i].end());
        combinedRow.insert(combinedRow.end(), matrix2[i].begin(), matrix2[i].end());

        // Add the combined row to the concatenated matrix
        concatenatedMatrix.push_back(combinedRow);
    }

    return concatenatedMatrix;
}

std::vector<double> crossProduct(const std::vector<double>& v1, const std::vector<double>& v2) {
    // Check if the input vectors have three elements
    if (v1.size() != 3 || v2.size() != 3) {
        std::cerr << "Error: Input vectors must be 3-dimensional." << std::endl;
        return {};
    }

    // Compute the cross product
    std::vector<double> result(3);
    result[0] = v1[1] * v2[2] - v1[2] * v2[1];
    result[1] = v1[2] * v2[0] - v1[0] * v2[2];
    result[2] = v1[0] * v2[1] - v1[1] * v2[0];
    return result;
}

// Function to extract a submatrix from a matrix
std::vector<std::vector<double>> extractSubmatrix(const std::vector<std::vector<double>>& matrix, int startRow, int endRow, int startCol, int endCol) {
    std::vector<std::vector<double>> submatrix;

    for (int i = startRow; i < endRow; ++i) {
        std::vector<double> row;
        for (int j = startCol; j < endCol; ++j) {
            row.push_back(matrix[i][j]);
        }
        submatrix.push_back(row);
    }

    return submatrix;
}

std::vector<std::vector<double>> subtractMatrices(const std::vector<std::vector<double>>& matrix1, const std::vector<std::vector<double>>& matrix2) {
    // Ensure matrices have the same dimensions
    if (matrix1.size() != matrix2.size() || matrix1[0].size() != matrix2[0].size()) {
        std::cerr << "Error: Matrices must have the same dimensions for subtraction." << std::endl;
        return {};
    }

    // Create a result matrix with the same dimensions as the input matrices
    std::vector<std::vector<double>> result(matrix1.size(), std::vector<double>(matrix1[0].size(), 0));

    // Subtract corresponding elements
    for (size_t i = 0; i < matrix1.size(); ++i) {
        for (size_t j = 0; j < matrix1[0].size(); ++j) {
            result[i][j] = matrix1[i][j] - matrix2[i][j];
        }
    }

    return result;
}

std::vector<std::vector<double>> addMatrices(const std::vector<std::vector<double>>& matrix1, const std::vector<std::vector<double>>& matrix2) {
    // Ensure matrices have the same dimensions
    if (matrix1.size() != matrix2.size() || matrix1[0].size() != matrix2[0].size()) {
        std::cerr << "Error: Matrices must have the same dimensions for subtraction." << std::endl;
        return {};
    }

    // Create a result matrix with the same dimensions as the input matrices
    std::vector<std::vector<double>> result(matrix1.size(), std::vector<double>(matrix1[0].size(), 0));

    // Subtract corresponding elements
    for (size_t i = 0; i < matrix1.size(); ++i) {
        for (size_t j = 0; j < matrix1[0].size(); ++j) {
            result[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }

    return result;
}

std::vector<double> DoubleVectorfunction(const std::vector<float>& floatVector) {
    std::vector<double> doubleVector;
    doubleVector.reserve(floatVector.size());  // Reserve space for efficiency

    for (const auto& value : floatVector) {
        doubleVector.push_back(static_cast<double>(value));
    }

    return doubleVector;
}

std::vector<std::vector<double>> convertToDoubleVector(const std::vector<std::vector<float>>& floatVector) {
    std::vector<std::vector<double>> doubleVector;
    for (const auto& row : floatVector) {
        std::vector<double> convertedRow;
        for (const auto& value : row) {
            if (std::isnan(value)) {
                convertedRow.push_back(std::numeric_limits<double>::quiet_NaN());
            }
            else {
                convertedRow.push_back(static_cast<double>(value));
            }
        }
        doubleVector.push_back(convertedRow);
    }
    return doubleVector;
}

// Define a function to create a meshgrid
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> createMeshgrid(
    const std::vector<std::vector<double>>& data,
    double pointSpacing) {
    // Determine the range
    double minX = data[0][0];
    double maxX = data[0][0];
    double minY = data[0][1];
    double maxY = data[0][1];

    for (const auto& point : data) {
        minX = std::min(minX, point[0]);
        maxX = std::max(maxX, point[0]);
        minY = std::min(minY, point[1]);
        maxY = std::max(maxY, point[1]);
    }

    // Create meshgrid
    std::vector<std::vector<double>> xq;
    std::vector<std::vector<double>> yq;

    for (double y = minY; y <= maxY; y += pointSpacing) {
        std::vector<double> rowXq;
        std::vector<double> rowYq;

        for (double x = minX; x <= maxX; x += pointSpacing) {
            rowXq.push_back(x);
            rowYq.push_back(y);
        }

        xq.push_back(rowXq);
        yq.push_back(rowYq);
    }

    return { xq, yq };
}


void extractXYVectors(const std::vector<std::vector<double>>& meshgrid,
    std::vector<double>& xq, bool isy) {
    if (meshgrid.empty() || meshgrid[0].empty()) {
        std::cerr << "Invalid meshgrid." << std::endl;
        return;
    }

    int numRows = meshgrid.size();
    int numCols = meshgrid[0].size();

    // Extract xq and yq vectors
    xq.clear();
    if (isy) {
        for (int i = 0; i < numRows; ++i) {
            for (int j = 0; j < numCols; ++j) {
                if (j == 1) {
                    // Assuming the second column represents y values
                    xq.push_back(meshgrid[i][j]);
                }
            }
        }
    }
    else if (isy == false) {
        xq = meshgrid[0];
    }
}

std::vector<std::vector<cv::Point2f>> convertToPoints(
    const std::vector<std::vector<double>>& xq,
    const std::vector<std::vector<double>>& yq) {
    // Ensure xq and yq have the same dimensions
    if (xq.size() != yq.size() || xq.empty() || xq[0].size() != yq[0].size()) {
        std::cerr << "Error: Input vectors have different dimensions." << std::endl;
        return std::vector<std::vector<cv::Point2f>>(); // Return an empty vector in case of error
    }

    // Create the points vector
    std::vector<std::vector<cv::Point2f>> points;

    // Iterate through the elements of xq and yq, creating Point2f objects
    for (size_t i = 0; i < xq.size(); ++i) {
        std::vector<cv::Point2f> row_points;
        for (size_t j = 0; j < xq[i].size(); ++j) {
            cv::Point2f point(static_cast<float>(xq[i][j]), static_cast<float>(yq[i][j]));
            row_points.push_back(point);
        }
        points.push_back(row_points);
    }

    return points;
}

std::vector<std::vector<cv::Point2f>> processMatrices(const std::vector<std::vector<double>>& dataMatrix,
    const std::vector<std::vector<cv::Point2f>>& coordinatesMatrix) {
    std::vector<std::vector<cv::Point2f>> resultMatrix;

    for (size_t i = 0; i < dataMatrix.size(); ++i) {
        std::vector<cv::Point2f> resultRow;
        for (size_t j = 0; j < dataMatrix[i].size(); ++j) {
            if (dataMatrix[i][j] == 1) {
                resultRow.push_back(coordinatesMatrix[i][j]);
            }
            else {
                resultRow.push_back(cv::Point2f(NAN, NAN));
            }
        }
        resultMatrix.push_back(resultRow);
    }

    return resultMatrix;
}

void replaceOnesWithIncrementedValues(const std::vector<std::vector<double>>& inputMatrix,
    std::vector<std::vector<float>>& outputMatrix,
    const std::vector<float>& resultvec) {
    int replacementValue = 0;
    for (size_t i = 0; i < inputMatrix.size(); ++i) {
        for (size_t j = 0; j < inputMatrix[i].size(); ++j) {
            if (std::isnan(inputMatrix[i][j])) {
                // Handle NaN values in inputMatrix
                outputMatrix[i][j] = std::numeric_limits<float>::quiet_NaN();
            }
            else if (inputMatrix[i][j] == 1) {
                if (replacementValue < resultvec.size()) {
                    outputMatrix[i][j] = resultvec[replacementValue];
                    ++replacementValue;
                }
                else {
                    outputMatrix[i][j] = 1;
                    std::cerr << "Error: replacementValue exceeds the total number of elements." << std::endl;
                }
            }
        }
    }
}

struct Point3D {
    float x;
    float y;
    float z;
};

// Custom hash function for std::pair<double, double>
struct PairHash {
    template <typename T1, typename T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);

        // Simple hash combining technique
        return h1 ^ h2;
    }
};

std::vector<std::vector<Point3D>> associateZValues(const std::vector<std::vector<double>>& printshape3,
    const std::vector<std::vector<cv::Point2f>>& coordinates2D) {

    // Create and initialize coordinates3D
    std::vector<std::vector<Point3D>> coordinates3D(coordinates2D.size(),
        std::vector<Point3D>(coordinates2D[0].size(),
            { 0.0f, 0.0f, -std::numeric_limits<float>::infinity() }));

    // Create a map for quick lookups with custom hash function
    std::unordered_multimap<std::pair<double, double>, float, PairHash> printshape3Map;
    for (const auto& entry : printshape3) {
        printshape3Map.emplace(std::make_pair(entry[0], entry[1]), static_cast<float>(entry[2]));
    }

    // Associate z values
    for (size_t i = 0; i < coordinates2D.size(); ++i) {
        for (size_t j = 0; j < coordinates2D[i].size(); ++j) {
            double x2D = coordinates2D[i][j].x;
            double y2D = coordinates2D[i][j].y;
            // Find corresponding z value in printshape3Map
            auto range = printshape3Map.equal_range({ x2D, y2D });
            float maxZValue = -std::numeric_limits<float>::infinity();

            for (auto it = range.first; it != range.second; ++it) {
                maxZValue = std::max(maxZValue, it->second);
            }

            // Convert and assign to Point3D
            coordinates3D[i][j] = { static_cast<float>(x2D), static_cast<float>(y2D), maxZValue };
            if (coordinates3D[i][j].z == maxZValue) {
                if (std::abs(x2D) < 1e-6 && std::abs(y2D) < 1e-6) {
                    coordinates3D[i][j] = { static_cast<float>(x2D), static_cast<float>(y2D), 0.0f };
                }
            }
        }
    }

    return coordinates3D;
}

// Custom function to replace -inf values in a row
void replaceInfValuesInRow(std::vector<cv::Point3f>& row) {
    float Infval = -std::numeric_limits<float>::infinity();
    std::vector<float> nonInfValues;

    // Collect non-infinite values in the row
    for (const auto& point : row) {
        if (point.z != Infval) {
            nonInfValues.push_back(point.z);
        }
    }

    // Replace -inf values based on the collected non-infinite values
    if (!nonInfValues.empty()) {
        float averageValue;
        if (nonInfValues.size() == 1) {
            averageValue = nonInfValues[0];
        }
        else if (nonInfValues.size() == 2) {
            averageValue = std::accumulate(nonInfValues.begin(), nonInfValues.end(), 0.0f) / 2;
        }

        for (auto& point : row) {
            if (point.z == Infval) {
                point.z = averageValue;
            }
        }
    }
}

// Function to replace -inf values in a matrix of Point3f
std::vector<std::vector<cv::Point3f>> replaceInfZValuesInMatrix(const std::vector<std::vector<cv::Point3f>>& coordinates3D) {
    std::vector<std::vector<cv::Point3f>> resultMatrix = coordinates3D;

    for (auto& row : resultMatrix) {
        replaceInfValuesInRow(row);
    }

    return resultMatrix;
}

std::pair<std::vector<double>, std::vector<double>> extractAndFlattenDimensions(std::vector<std::vector<cv::Point2f>>& coordloc) {
    // Initialize vectors for x and y dimensions
    std::vector<double> xCoordinates, yCoordinates;

    // Iterate through each row (vector of Point2f)
    for (auto& row : coordloc) {
        // Iterate through each Point2f in the current row
        for (auto it = row.begin(); it != row.end(); ) {
            // Check for (nan, nan) and erase if found
            if (std::isnan(it->x) || std::isnan(it->y)) {
                it = row.erase(it);
            }
            else {
                xCoordinates.push_back(static_cast<double>(it->x));
                yCoordinates.push_back(static_cast<double>(it->y));
                ++it;
            }
        }
    }

    return std::make_pair(xCoordinates, yCoordinates);
}

// Function to convert a 2D vector to a vector of arrays and select columns
std::vector<std::array<double, 3>> convertAndSelectColumns(const std::vector<std::vector<double>>& input, int col1, int col2, int col3) {
    std::vector<std::array<double, 3>> result;

    for (const auto& row : input) {
        if (row.size() >= 3) {
            std::array<double, 3> newArray;
            newArray[0] = (col1 >= 0 && col1 < row.size()) ? row[col1] : 0.0;
            newArray[1] = (col2 >= 0 && col2 < row.size()) ? row[col2] : 0.0;
            newArray[2] = (col3 >= 0 && col3 < row.size()) ? row[col3] : 0.0;
            result.push_back(newArray);
        }
        else {
            // Handle the case where the row doesn't have enough elements
            std::cerr << "Warning: Skipping row with insufficient elements." << std::endl;
        }
    }

    return result;
}

std::vector<double> extractColumn(const std::vector<std::vector<double>>& matrix, size_t column) {
    std::vector<double> result;

    for (const auto& row : matrix) {
        if (column < row.size()) {
            result.push_back(row[column]);
        } else {
            // Handle the case where the column index is out of bounds for some rows
            // You may choose to throw an exception or handle it differently based on your requirements
            // For simplicity, this example just pushes back 0.0 in such cases
            result.push_back(0.0);
        }
    }

    return result;
}

std::vector<std::vector<cv::Point3f>> convertToCvPoint3f(const std::vector<std::vector<Point3D>>& input) {
    std::vector<std::vector<cv::Point3f>> output;

    for (const auto& row : input) {
        std::vector<cv::Point3f> convertedRow;
        for (const auto& point : row) {
            convertedRow.push_back(cv::Point3f(point.x, point.y, point.z));
        }
        output.push_back(convertedRow);
    }

    return output;
}

float triangle_area(float x1, float y1, float x2, float y2, float x3, float y3) {
    // Calculate the area of the triangle using the Shoelace Formula
    float area = static_cast<float>( 0.5 * std::fabs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)));
    return area;
}

float distance(float x1, float y1, float x2, float y2) {
    float dist = static_cast<float>(std::sqrt(std::pow(x1 - x2, 2) + std::pow(y1 - y2, 2)));
    return dist;
}

void interpolation(std::vector<std::vector<cv::Point3f>>& matrix1, std::vector<cv::Point2f>& points, std::vector<float>& interpolated) {
    for (int i = 0; i < points.size(); ++i) {
        float P1[3];
        float P2[3];
        float P3[3];
        float P[2];
        P[0] = points[i].x;
        P[1] = points[i].y;
        P1[0] = matrix1[i][0].x;
        P1[1] = matrix1[i][0].y;
        P1[2] = matrix1[i][0].z;
        P2[0] = matrix1[i][1].x;
        P2[1] = matrix1[i][1].y;
        P2[2] = matrix1[i][1].z;
        P3[0] = matrix1[i][2].x;
        P3[1] = matrix1[i][2].y;
        P3[2] = matrix1[i][2].z;
        // there is nothing we can do here, just put z value into v
        float v;
        if ((P3[2] == 0 and P2[2] == 0 and P1[2] == 0) and
            (P3[1] == 0 and P2[1] == 0 and P1[1] == 0) and
            (P3[0] == 0 and P2[0] == 0 and P1[0] == 0)) {
            v = 0.0f;
        }
        if ((P3[2] == 0 and P2[2] == 0 and P1[2] != 0) and
            (P3[1] == 0 and P2[1] == 0 and P1[1] != 0) and
            (P3[0] == 0 and P2[0] == 0 and P1[0] != 0)) {
            v = P1[2];
        }
        else if ((P3[2] == 0 and P2[2] != 0 and P1[2] == 0) and
            (P3[1] == 0 and P2[1] != 0 and P1[1] == 0) and
            (P3[0] == 0 and P2[0] != 0 and P1[0] == 0)) {
            v = P2[2];
        }
        else if ((P3[2] != 0 and P2[2] == 0 and P1[2] == 0) and
            (P3[1] != 0 and P2[1] == 0 and P1[1] == 0) and
            (P3[0] != 0 and P2[0] == 0 and P1[0] == 0)) {
            v = P3[2];
        }
        //linear interpolation in 1D
        else if ((P3[2] == 0 and P2[2] != 0 and P1[2] != 0) and
            (P3[1] == 0 and P2[1] != 0 and P1[1] != 0) and
            (P3[0] == 0 and P2[0] != 0 and P1[0] != 0)) {
            v = ((distance(P[0], P[1], P1[0], P1[1]) * P1[2]) * distance(P[0], P[1], P2[0], P2[1]) * P2[2]) / distance(P1[0], P1[1], P2[0], P2[1]);
        }
        else if ((P3[2] != 0 and P2[2] == 0 and P1[2] != 0) and
            (P3[1] != 0 and P2[1] == 0 and P1[1] != 0) and
            (P3[0] != 0 and P2[0] == 0 and P1[0] != 0)) {
            v = ((distance(P[0], P[1], P1[0], P1[1]) * P1[2]) * distance(P[0], P[1], P3[0], P3[1]) * P3[2]) / distance(P1[0], P1[1], P3[0], P3[1]);
        }
        else if ((P3[2] != 0 and P2[2] != 0 and P1[2] == 0) and
            (P3[1] != 0 and P2[1] != 0 and P1[1] == 0) and
            (P3[0] != 0 and P2[0] != 0 and P1[0] == 0)) {
            v = ((distance(P[0], P[1], P2[0], P2[1]) * P2[2]) * distance(P[0], P[1], P3[0], P3[1]) * P3[2]) / distance(P2[0], P2[1], P3[0], P3[1]);
        }
        //triangular base linear interpolation in 2D
        else if ((P3[2] != 0 and P2[2] != 0 and P1[2] != 0) and
            (P3[1] != 0 and P2[1] != 0 and P1[1] != 0) and
            (P3[0] != 0 and P2[0] != 0 and P1[0] != 0)) {
            float abc, pbc, apc, abp;
            abc = triangle_area(P1[0], P1[1], P2[0], P2[1], P3[0], P3[1]);
            pbc = triangle_area(P[0], P[1], P2[0], P2[1], P3[0], P3[1]);
            apc = triangle_area(P1[0], P1[1], P[0], P[1], P3[0], P3[1]);
            abp = triangle_area(P1[0], P1[1], P2[0], P2[1], P[0], P[1]);
            v = (pbc * P1[2] + apc * P2[2] + abp * P3[2]) / abc;
        }
        interpolated.push_back(v);
    }
}

std::vector<std::vector<float>> pfillMatrices(const std::vector<std::vector<double>>& dataMatrix,
    std::vector<float>& values) {
    std::vector<std::vector<float>> resultMatrix;
    int ij = 0;
    for (size_t i = 0; i < dataMatrix.size(); ++i) {
        std::vector<float> resultRow;
        for (size_t j = 0; j < dataMatrix[i].size(); ++j) {
            if (dataMatrix[i][j] == 1) {
                if (ij < values.size()) {
                    resultRow.push_back(values[ij]);
                    ++ij;
                }
                else {
                    resultRow.push_back(NAN);
                }
            }
            else {
                resultRow.push_back(NAN);
            }
        }
        resultMatrix.push_back(resultRow);
    }

    return resultMatrix;
}

void griddata(std::vector<std::vector<double>>& inputmat,
    std::vector<std::vector<double>>& xq,
    std::vector<std::vector<double>>& yq,
    std::vector<std::vector<double>>& zq,
    std::vector<double>& ShapeX,
    std::vector<double>& ShapeY,
    std::vector<double>& ShapeZ,
    double pointSpacing)
{
    std::vector<std::vector<double>> printshape3;
    std::vector <double> thirdrawprintshape, firstrawprintshape, secondrawprintshape;
    // Iterate through the original vector and extract the first three columns
    for (const auto& row : inputmat) {
        if (row.size() >= 3) {
            printshape3.push_back({ row[0], row[1], row[2] });
            firstrawprintshape.push_back(row[0]);
            secondrawprintshape.push_back(row[1]);
            thirdrawprintshape.push_back(row[2]);

        }
        else {
            // Handle the case where the row has fewer than three columns
            std::cerr << "Warning: Row has fewer than three columns." << std::endl;
        }
    }
    auto meshgrid = createMeshgrid(inputmat, pointSpacing);

    xq = meshgrid.first;
    yq = meshgrid.second;

    std::vector<std::vector<cv::Point2f>> xqyqpoints = convertToPoints(xq, yq);
    std::vector<double> vectxq, vectyq;
    extractXYVectors(xq, vectxq, false);
    extractXYVectors(yq, vectyq, true);
    // Construct vector of cv::Point2f
    std::vector<cv::Point2f> gridpoints;
    for (size_t i = 0; i < firstrawprintshape.size(); ++i) {
        float x_valeue = static_cast<float>(firstrawprintshape[i]);
        float y_valeue = static_cast<float>(secondrawprintshape[i]);
        cv::Point2f pointi(x_valeue, y_valeue);
        gridpoints.push_back(pointi);
    }

    // Find the convex hull of the points
    std::vector<cv::Point2f> hull;
    cv::convexHull(gridpoints, hull);
    // Now, 'hull' contains the contour points of the geometrical figure

    std::vector<std::vector<double>> dataloca;

    // Iterate through the vectors of points and check if each point is inside the shape
    for (const auto& points : xqyqpoints) {
        std::vector<double> resultRow;
        for (const auto& point : points) {
            double distance = cv::pointPolygonTest(hull, point, true);

            if (distance > 0) {
                // The point is inside the shape
                resultRow.push_back(1.0);
            }
            else if (distance == 0) {
                // The point is on the contour
                resultRow.push_back(0.0);
            }
            else {
                // The point is outside the shape
                resultRow.push_back(NaN);
            }
        }
        dataloca.push_back(resultRow);
    }
    std::vector<std::vector<cv::Point2f>> coordloc = processMatrices(dataloca, xqyqpoints);
    //see in which delaunay triangle the point is
    cv::Rect boundingRect = cv::boundingRect(gridpoints);

    // Create a Subdiv2D object and insert the points
    cv::Subdiv2D subdiv(boundingRect);
    subdiv.insert(gridpoints);
    std::vector<cv::Vec4f> edgeList;
    subdiv.getEdgeList(edgeList);
    std::vector<cv::Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);
    std::vector<std::vector<cv::Point2f>> resultvertices;
    std::vector<cv::Point2f> pt;
    for (const auto& points : coordloc) {
        for (const auto& point : points) {
            // Check for NaN values and skip them
            if (!std::isnan(point.x) && !std::isnan(point.y)) {
                // Find the corresponding triangle using locate
                int edgeIndex, vertex;
                subdiv.locate(point, edgeIndex, vertex);
                cv::Vec6f spectriangle;
                int nextEdge = subdiv.getEdge(edgeIndex, cv::Subdiv2D::NEXT_AROUND_ORG);
                cv::Point2f vertices[3];
                std::vector<cv::Point2f> verticesVector;
                if (nextEdge > 0) {
                    for (int i = 0; i < 3; ++i) {
                        int org = subdiv.edgeOrg(nextEdge, &vertices[i]);
                        nextEdge = subdiv.edgeDst(subdiv.nextEdge(nextEdge));
                        verticesVector.push_back(vertices[i]);
                    }
                    double distance = cv::pointPolygonTest(verticesVector, point, true);
                    if (distance > 0) {
                        resultvertices.push_back(verticesVector);
                        pt.push_back(point);
                    }
                }

            }
        }
    }
    auto ShapexandY = extractAndFlattenDimensions(coordloc);
    ShapeX = ShapexandY.first;
    ShapeY = ShapexandY.second;
    std::vector<std::vector<Point3D>> coordinates3D = associateZValues(printshape3, resultvertices);
    std::vector<std::vector<cv::Point3f>> convertedCoordinates3D = convertToCvPoint3f(coordinates3D);
    std::vector<std::vector<cv::Point3f>> modified3DMatrix = replaceInfZValuesInMatrix(convertedCoordinates3D);
    std::vector<float> interpolatedvals;
    interpolation(modified3DMatrix, pt, interpolatedvals);
    std::vector<std::vector<float>> filledmat = pfillMatrices(dataloca, interpolatedvals);
    zq = convertToDoubleVector(filledmat);
    ShapeZ = DoubleVectorfunction(interpolatedvals);
}


// Custom comparison function for cv::Point2f
bool pointComparator(const cv::Point2f& p1, const cv::Point2f& p2) {
    return ((p1.x < p2.x) && (p1.y == p2.y)) || ((p1.x == p2.x) && (p1.y < p2.y));
}

// Function to remove duplicates from a vector of cv::Point2f without touching data IDs
void removeDuplicates(std::vector<cv::Point2f>& points) {
    // Sort the vector using the custom comparison function
    std::sort(points.begin(), points.end(), pointComparator);

    // Use std::unique to rearrange the vector and get the iterator to the new end
    auto newEnd = std::unique(points.begin(), points.end(), pointComparator);

    // Erase the duplicates from the vector
    points.erase(newEnd, points.end());
}

void STLSURF(const std::vector<double>& X, const std::vector<double>& Y, const std::vector<std::array<double, 3>>& centroids, const std::vector<std::array<double, 3>>& normals,
    std::vector<double>& Z, std::vector<double>& nX, std::vector<double>& nY, std::vector<double>& nZ) {
    if (X.size() != Y.size()) {
        throw std::runtime_error("First two arguments must have the same length");
    }

    if (centroids.empty() || normals.empty()) {
        throw std::runtime_error("Centroids and normals vectors must not be empty");
    }

    Z.resize(X.size());
    nX.resize(X.size());
    nY.resize(X.size());
    nZ.resize(X.size());

    for (size_t i = 0; i < X.size(); ++i) {
        double minDistance = std::numeric_limits<double>::max();
        size_t nearestCentroidIndex = 0;

        for (size_t j = 0; j < centroids.size(); ++j) {
            const auto& centroid = centroids[j];
            double distanceSquared = std::pow(X[i] - centroid[0], 2) + std::pow(Y[i] - centroid[1], 2);

            if (distanceSquared < minDistance) {
                minDistance = distanceSquared;
                nearestCentroidIndex = j;
            }
        }

        const auto& normalsForNearestPoint = normals[nearestCentroidIndex];

        nX[i] = normalsForNearestPoint[0];
        nY[i] = normalsForNearestPoint[1];
        nZ[i] = normalsForNearestPoint[2];

        double planeDistance = nX[i] * centroids[nearestCentroidIndex][0] + nY[i] * centroids[nearestCentroidIndex][1] + nZ[i] * centroids[nearestCentroidIndex][2];
        Z[i] = (planeDistance - nX[i] * X[i] - nY[i] * Y[i]) / nZ[i];
    }
}

void runningAverage(const std::vector<std::vector<double>>& inpath, int num, std::vector<std::vector<double>>& outpath) {
    int numRows = inpath.size();
    int numCols = inpath[0].size();
    // Initialize outpath with zeros
    outpath = std::vector<std::vector<double>>((numCols - num + 1), std::vector<double>(numCols, 0.0));
    for (int i = 0; i < num; ++i) {
        for (int j = 0; j < numRows - num + 1; ++j) {
            for (int k = 0; k < numCols; ++k) {
                outpath[j][k] += inpath[i + j][k];
            }
        }
    }

    for (int i = 0; i < numRows - num + 1; ++i) {
        for (int j = 0; j < numCols; ++j) {
            outpath[i][j] /= num;
        }
    }
}

void sort_column(std::vector<std::vector<double>>& matrix, size_t column)
{
    std::sort(matrix.begin(), matrix.end(),
              [column](const std::vector<double>& a, const std::vector<double>& b)
              { return a[column] < b[column]; });
}

std::vector<std::vector<double>> calculateEquivalent(const std::vector<std::vector<double>>& initialpath,
                                                      float& linespacing,
                                                      const std::vector<std::vector<double>>& sidevec,
                                                      bool negative) {
    std::vector<std::vector<double>> result;
    int endfor;
    if (initialpath.size() - 1 > sidevec.size()-1){
        endfor = sidevec.size()-1;
    } else {
        endfor = initialpath.size() -1;
    }
    for (size_t i = 0; i < endfor; ++i) {
        std::vector<double> row;
        for (size_t j = 0; j < 3; ++j) {
            if (negative == true){
                double val = initialpath[i][j] -
                         static_cast<double>(linespacing) * sidevec[i][j] /
                         sqrt(sidevec[i][0] * sidevec[i][0] +
                              sidevec[i][1] * sidevec[i][1] +
                              sidevec[i][2] * sidevec[i][2]);
                row.push_back(val);
            }else{
                double val = initialpath[i][j] +
                         static_cast<double>(linespacing) * sidevec[i][j] /
                         sqrt(sidevec[i][0] * sidevec[i][0] +
                              sidevec[i][1] * sidevec[i][1] +
                              sidevec[i][2] * sidevec[i][2]);
                row.push_back(val);
            }
        }
        result.push_back(row);
    }

    return result;
}

std::vector<std::vector<double>> stretching_path (std::vector<std::vector<double>>& newpath, std::vector<std::vector<double>>& contour, bool forward) {
    std::vector<std::vector<double>> result;
    double offset;
    std::vector<std::vector<double>> linepoints;
    std::vector<double> linediff;
    if (forward == true) {
        linepoints = extractSubmatrix(newpath, newpath.size()-1,newpath.size(),0,2);
        //std::cout << linepoints.size() << std::endl;
        for (int i=0; i<linepoints.size(); i++) {
            linediff.push_back(linepoints[i][1] - linepoints[i][0] );
        }
        offset = newpath[newpath.size()-1][3];
        offset = offset -0;
    } else {
        linepoints = extractSubmatrix(newpath, 0,1 ,0,2);
        //std::cout << linepoints.size() << std::endl;
        for (int i=0; i<linepoints.size(); i++) {
            linediff.push_back(linepoints[i][0] - linepoints[i][1] );
        }
        offset = newpath[0][3];
        offset = offset -0;
    }
    double scale;
    double abslinediff=0.0;
    for (int i=0; i<linediff.size(); i++){
        abslinediff = abslinediff+ linepoints[i][3];
    }
    abslinediff=abslinediff/(linediff.size()-1);
    if (abslinediff<0){
        abslinediff=-abslinediff;
    }
    scale=offset/abslinediff;
    std::vector<double> line1;
    std::vector<std::vector<double>> endpoint;

    if (forward == true) {line1 = extractColumn(linepoints,0);}else {
    line1 = extractColumn(linepoints,1);}

    for (int i=0; i<linediff.size(); i++){
        std::vector<double> line2;
        line2.push_back( (scale*linediff[i])+line1[i]);
        endpoint.push_back(line2);
        //std::cout <<endpoint[i] << std::endl;
    }

    std::vector<double> addZ;
    std::vector<double> addnX;
    std::vector<double> addnY;
    std::vector<double> addnZ;
    STLSURF(extractColumn(endpoint,0), extractColumn(endpoint,1), convertAndSelectColumns(contour, 0, 1, 2), convertAndSelectColumns(contour, 3, 4, 5), addZ, addnX, addnY, addnZ);
    std::vector<std::vector<double>> newline;
    newline = {
            extractColumn(endpoint,0), extractColumn(endpoint,1), addZ, addnX, addnY, addnZ
        };

    result = newpath;
    result.push_back(newline[0]);
    return result;
}

// Function to insert a vector into a matrix at a specified position
void insertVectorIntoMatrix(vector<vector<double>>& matrix, const vector<double>& vec, int row, int col) {
    // Check if the dimensions match
    if (matrix.size() < row + 1 || matrix[0].size() < col + vec.size()) {
        cout << "Invalid dimensions for insertion!" << endl;
        return;
    }

    // Insert the vector into the matrix
    for (int i = 0; i < vec.size(); ++i) {
        matrix[row][col + i] = vec[i];
    }
}

vector<double> calculatePathD(const vector<vector<double>>& newpathR) {
    vector<double> pathD;
    for (size_t i = 0; i < newpathR.size() - 1; ++i) {
        double distance = sqrt(pow(newpathR[i + 1][0] - newpathR[i][0], 2) +
                               pow(newpathR[i + 1][1] - newpathR[i][1], 2) +
                               pow(newpathR[i + 1][2] - newpathR[i][2], 2));
        //std:: cout << newpathR[i + 1][0] << "" << newpathR[i][0] << std::endl;
        pathD.push_back(distance);
    }

    // Append a zero at the end
    pathD.push_back(0);
    return pathD;
}

// Function to subdivide path segments to ensure no segment exceeds a certain step size
vector<vector<double>> subdividePath(vector<vector<double>> newpathR, double stepsize) {
    vector<double> pathD = calculatePathD(newpathR);
    vector<vector<double>> newpath;

    // Continue until all segments are within the step size
    while (true) {
        bool exceed = false;
        vector<size_t> skipI;

        // Check segments longer than twice the step size
        for (size_t i = 0; i < pathD.size(); ++i) {
            if (pathD[i] > stepsize * 2) {
                exceed = true;
                skipI.push_back(i);
            }
        }

        // Exit loop if no segments exceed the step size
        if (!exceed) break;
        //std::cout << skipI.size() << std::endl;
        // Insert midpoints for long segments
        for (size_t i = 0; i < skipI.size(); ++i) {
            size_t index = skipI[i];
            vector<double> newrow(3);
            for (int j = 0; j < 3; ++j) {
                newrow[j] = (newpathR[index][j] + newpathR[index+1][j]) / 2.0;
            }
            insertVectorIntoMatrix(newpathR,newrow,index,0);
        }

        // Recalculate distances
        pathD = calculatePathD(newpathR);
    }

    return newpathR;
}

std::vector<std::vector<double>> filterInitialPath(std::vector<std::vector<double>>& initialpath, std::vector<cv::Point2f>& countourpoints) {
    std::vector<std::vector<double>> verifiedinitialpath;
    std::vector<cv::Point2f> querryshapepoints;

    for (size_t i = 0; i < initialpath[0].size(); ++i) {
        float x_qpoint = static_cast<float>(initialpath[0][i]);
        float y_qpoint = static_cast<float>(initialpath[1][i]);
        cv::Point2f qpoints(x_qpoint, y_qpoint);
        querryshapepoints.push_back(qpoints);
    }

    std::vector<int> indices;
    for (size_t i = 0; i < querryshapepoints.size(); ++i) {
        if (cv::pointPolygonTest(countourpoints, querryshapepoints[i], true) >= 0) {
            indices.push_back(static_cast<int>(i));
        }
    }

    for (size_t i = 0; i < indices.size(); ++i) {
        std::vector<double> transitionvector;
        for (size_t j = 0; j < initialpath.size(); ++j) {
            transitionvector.push_back(initialpath[j][indices[i]]);
        }
        verifiedinitialpath.push_back(transitionvector);
    }

    initialpath = verifiedinitialpath;
    return initialpath;
}


void SLICER_CALCULATIONS(
    std::vector<std::vector<double>>& points,
    std::vector<std::vector<double>>& Nvecs,
    std::vector<std::vector<double>>& nvecpoints)
{
    float stepsize_contour = 0.13f; //sampling distance between points for contoured infill, mm
    float stepsize_contour_border = 0.05f; //sampling distance between points for contoured borders, mm

    float layerheight = 0.3f; //non-contoured layer height, mm (for planar layers)
    float linewidth = 0.4f; //nozzlewidth, mm
    float close_linespacing = 0.4f; //spacing between close contoured lines, mm (not upper contours)

    float support_interface_offset = 0.3f; //Gap between support and upper contoured layers (mm)
    float sampling_dist = 0.01f; //final point spacing for contoured lines, mm (interpolated at end of generation)

    float upper_layers_flowfactor = 3.3f; //flow rate multiplier for upper contoured layers
    float upper_layers_borderfactor = 4.0f; //flow rate multiplier for upper contoured layer borders

    float flowfactor = 1.3f; //flow rate multiplier for all other layers

    bool stretch_down = true; //True: stretch paths toward the edge of the XY-coordinate-region of the part, False: stretch paths toward the bottom

    bool clip_paths_every_iteration = true; //true: trim paths every iteration (slower, but less errors), false: trim paths after all iterations (faster)

    double support_temp = 110; //support material extruder temperature (°C)
    double mesh_temp = 93.334; //upper layer material extruder temperature (°C)
    support_temp = this_is_america(support_temp);
    mesh_temp = this_is_america(mesh_temp);

    float topcontour_linespacing = 1.2f; //upper layers spacing between paths (mm)
    float num = 25.0f; //number of samples for running average smoothing (more = more smooth, less accurate)
    float filamentD = 1.75f; //filament diameter (mm)

    int infillspacing = 3; //linespacing for infill/support material mm
    int skinlayer = 0; //number of layers of outer skin for the support material/planar layers
    int wall_lines = 1; //number of wall lines for the support material/planar layers
    int wallsmoothnum = 27; //number of samples for running average smoothing of walls for the support/planar layers  (more = more smooth, less accurate)

    bool flatbottom = true; //true: part sits flat on build plate, false: part is upper layers only (use for mesh lens)
    int bordersmoothnum = 40; //number of samples for running average smoothing for contoured borders (more = more smooth, less accurate)
    int contourborderlines = 3; //number of contoured border lines

    float contourlayerheight = 0.2f; //contoured layers layer height, mm
    float contourthickness = 4.0f; //total contoured thickness, mm
    float num_contourlayers; //number of contoured layers
    num_contourlayers = contourthickness / contourlayerheight;
    int INTnum_contourlayers = static_cast<int>(std::ceil(num_contourlayers));
    int num_topcontour = 2; //number of upper contoured layers (not support)
    int num_topborder = 2; //number of border lines in the upper contoured layers

    double middleX = calculateMean(points, 0);
    double middleY = calculateMean(points, 1);
    double stretchFX = calculateMax(points, 0) + 4.0;
    double stretchFY = calculateMax(points, 1) + 4.0;
    double stretchBX = calculateMin(points, 0) - 4.0;
    double stretchBY = calculateMin(points, 1) - 4.0;
    double clearZ = calculateMax(points, 2) + 4.0;
    std::vector<std::vector<double>> lims = { {(stretchBX + 4.0), (stretchFX - 4.0)},
                                            {(stretchBY + 4.0), (stretchFY - 4.0)} };
    std::vector<std::vector<std::vector<double>>> contourlayers;
    std::vector<std::vector<double>> layers(INTnum_contourlayers, std::vector<double>(1));
    std::vector<std::vector<double>> Lpoints(nvecpoints.size(), std::vector<double>(nvecpoints[0].size(), 0));
    std::vector<std::vector<double>> Lnegative(nvecpoints.size(), std::vector<double>(nvecpoints[0].size(), 0));
    for (int k = 0; k <= INTnum_contourlayers; ++k) {
        for (size_t i = 0; i < nvecpoints.size(); ++i) {
            for (size_t j = 0; j < nvecpoints[0].size(); ++j) {
                Lnegative[i][j] = Nvecs[i][j] * k * contourlayerheight;
                Lpoints[i][j] = nvecpoints[i][j] - Lnegative[i][j];
            }
        }
        contourlayers.push_back(concatenateMatrices(Lpoints, Nvecs));
    }
    float pointspacing;
    pointspacing = stepsize_contour_border;
    // Call the function to get the first 2D matrix
    std::vector<std::vector<double>> printshape = contourlayers[0];
    std::vector<std::vector<double>> xq, yq, zq;
    std::vector<double> ShapeX, ShapeY, ShapeZ;
    griddata(printshape, xq, yq, zq, ShapeX, ShapeY, ShapeZ, pointspacing);
    // Construct vector of cv::Point2f
    std::vector<cv::Point2f> XYBIpoints;
    for (size_t i = 0; i < ShapeX.size(); ++i) {
        float x_valeue = static_cast<float>(ShapeX[i]);
        float y_valeue = static_cast<float>(ShapeY[i]);
        cv::Point2f pointi(x_valeue, y_valeue);
        XYBIpoints.push_back(pointi);
    }

    // Find the convex hull of the points
    std::vector<cv::Point2f> hullPoint;
    cv::convexHull(XYBIpoints, hullPoint);
    std::vector<cv::Point2f> countourpoints;

    // Extract points on the convex hull with distance 0
    for (const cv::Point2f& hullPoints : hullPoint) {
        for (size_t i = 0; i < XYBIpoints.size(); ++i) {
            float distance = static_cast<float>( cv::norm(hullPoints - XYBIpoints[i]));

            // Consider points with distance 1 like the matlab code
            if ((distance <= 1.0f) and (distance >= 0.0f)) {  // Adjust the tolerance as needed
                countourpoints.push_back(hullPoints);
            }
        }
    }
    removeDuplicates(countourpoints);
    std::vector<std::vector<cv::Point2f>> ccountourpoints;
    ccountourpoints.push_back(countourpoints);
    extractAndFlattenDimensions(ccountourpoints);
    auto YandXshapebound = extractAndFlattenDimensions(ccountourpoints);
    std::vector<double> Xshapebound, Yshapebound;
    Xshapebound = YandXshapebound.first;
    Yshapebound = YandXshapebound.second;
    std::cout << Yshapebound.size() << " Contours done !" << std::endl;
    bool Rstop_cond = false;
    bool Lstop_cond = false;
    float stepsize = stepsize_contour;
    float linespacing;
    int sortdimension;
    int perpdimension;
    for (int i = 1; i <= num_contourlayers; ++i) {
        if (i > (num_contourlayers - num_topcontour)) {
            linespacing = close_linespacing;
        }
        else {
            linespacing = topcontour_linespacing;
        }
        // Assuming i is a valid index
        std::vector<std::vector<double>> contour = contourlayers[i - 1];
        double stretchF;
        double stretchB;
        if (i % 2 == 1) {
            stretchF = stretchFY;
            stretchB = stretchBY;
            sortdimension = 1;
            perpdimension = 0;
        }
        else {
            stretchF = stretchFX;
            stretchB = stretchBX;
            sortdimension = 0;
            perpdimension = 1;
        }
        // Calculate initnumline
        std::vector<double> initnumline;
        for (double value = static_cast<double>(calculateMin(contour, perpdimension));
            value <= static_cast<double>(calculateMax(contour, perpdimension));
            value += static_cast<double>(stepsize)) {
            initnumline.push_back(value);
        }
        double avg_perp = (calculateMin(contour, perpdimension) + calculateMax(contour, perpdimension)) / 2.0;
        std::vector<double> otherdim(initnumline.size(), avg_perp);

        std::vector<double> initZ;
        std::vector<double> initnX;
        std::vector<double> initnY;
        std::vector<double> initnZ;
        std::vector<std::vector<double>> initialpath;
        if (i % 2 == 1) {
            STLSURF(otherdim, initnumline, convertAndSelectColumns(contour, 0, 1, 2), convertAndSelectColumns(contour, 3, 4, 5), initZ, initnX, initnY, initnZ);
            initialpath = {
                otherdim, initnumline, initZ, initnX, initnY, initnZ
            };
        }
        else {
            STLSURF(initnumline, otherdim, convertAndSelectColumns(contour, 0, 1, 2), convertAndSelectColumns(contour, 3, 4, 5), initZ, initnX, initnY, initnZ);
            initialpath = {
                initnumline, otherdim, initZ, initnX, initnY, initnZ
            };
        }
        // Sorting the rows of the matrix based on the specified column
        sort_column(initialpath,sortdimension);
        //initialpath, static_cast<int>(num)
        std::vector<std::vector<double>> averagedval;
        runningAverage(initialpath, static_cast<int>(num), averagedval);

        if (clip_paths_every_iteration == true) {
            initialpath = filterInitialPath(initialpath, countourpoints);
        }
        if (initialpath.empty() != true and initialpath.size() > static_cast<int>(num) ) {
            std::cout << "For layer nb " << i <<std::endl;
            std::cout << "found " << initialpath.size() << " initial paths"<< std::endl;

            // Extract submatrix initialpath(2:end, 1:3)
            std::vector<std::vector<double>> submatrix1 = extractSubmatrix(initialpath, 1, initialpath.size(), 0, 2);
            // Extract submatrix initialpath(1:end-1, 1:3)
            std::vector<std::vector<double>> submatrix2 = extractSubmatrix(initialpath, 0, initialpath.size()-1, 0, 2);
            // Extract submatrix initialpath(1:end-1, 4:6)
            std::vector<std::vector<double>> submatrix3 = extractSubmatrix(initialpath, 0, initialpath.size()-1, 3, 6);
            std::vector<std::vector<double>> substracted_matrix = subtractMatrices(submatrix1, submatrix2);
            //std::cout << submatrix1.size() <<" "<< submatrix2.size() <<" "<< submatrix3.size() << std::endl;
            std::vector<std::vector<double>> sidevec;
            for(int j = 0; j <= substracted_matrix.size(); ++j){
                std::vector<double> sbmat;
                std::vector<double> samat;
                std::vector<double> endamat;
                for (int k = 0; k <= substracted_matrix[0].size(); ++k){
                    sbmat.push_back(substracted_matrix[k][j]);
                    samat.push_back(submatrix3[k][j]);
                }
                endamat = crossProduct(sbmat, samat);
                if (endamat[0] != NaN &&  endamat[1] != NaN &&  endamat[2] != NaN){
                    if (static_cast<double>(abs(endamat[0]))  == 0.0 and  static_cast<double>(abs(endamat[1]))  == 0.0 and static_cast<double>(abs(endamat[2]))  == 0.0 ){
                    }else{
                        sidevec.push_back(endamat);
                        //std::cout << static_cast<double>(abs(endamat[0])) << " " <<  static_cast<double>(abs(endamat[1])) << " " << static_cast<double>(abs(endamat[2]))   << std::endl;
                        //std::cout << endamat.size() << " " << endamat[0] << " " << endamat[1] << " " << endamat[2] << std::endl;
                    }
                }
                //std::cout << endamat.size() << " " << endamat[0] << " " << endamat[1] << " " << endamat[2] << std::endl;
            }
            //std::cout << sidevec.size() << std::endl;
            std::vector<std::vector<double>> sidepointsR = calculateEquivalent(initialpath, linespacing, sidevec, false);
            std::vector<std::vector<double>> sidepointsL = calculateEquivalent(initialpath, linespacing, sidevec, true);
            //std::cout << sidepointsR.size() << " " << sidepointsR[0].size() << std::endl;
            //std::cout << sidepointsL.size() << " " << sidepointsL[0].size() << std::endl;
            std::vector<double> RZ;
            std::vector<double> RnX;
            std::vector<double> RnY;
            std::vector<double> RnZ;
            STLSURF(extractColumn(sidepointsR,0), extractColumn(sidepointsR,1), convertAndSelectColumns(contour, 0, 1, 2), convertAndSelectColumns(contour, 3, 4, 5), RZ, RnX, RnY, RnZ);
            std::vector<double> LZ;
            std::vector<double> LnX;
            std::vector<double> LnY;
            std::vector<double> LnZ;
            STLSURF(extractColumn(sidepointsL,0), extractColumn(sidepointsL,1), convertAndSelectColumns(contour, 0, 1, 2), convertAndSelectColumns(contour, 3, 4, 5), LZ, LnX, LnY, LnZ);
            std::vector<std::vector<double>> newpathR;
            newpathR = {
                extractColumn(sidepointsR,0), extractColumn(sidepointsR,1), RZ, RnX, RnY, RnZ
            };
            std::vector<std::vector<double>> newpathL;
            newpathL = {
                extractColumn(sidepointsL,0), extractColumn(sidepointsL,1), LZ, LnX, LnY, LnZ
            };
            sort_column(newpathR,sortdimension);
            std::vector<std::vector<double>> Raveragedval;
            //dodge segmentfault ^^
            int newnum = static_cast<int>(num);
            if (num> newpathR.size()){
                newnum = newpathR.size();
            }
            runningAverage(newpathR, newnum, Raveragedval);
            //std:: cout << newpathR[ 1][0] << "" << newpathR[0][0] << std::endl;
            sort_column(newpathL,sortdimension);
            std::vector<std::vector<double>> Laveragedval;
            runningAverage(newpathL, newnum, Laveragedval);
            //newpathR= stretching_path(newpathR,contour,false);
            if (stretch_down = true){
                if (Rstop_cond = false){
                    if (newpathR[newpathR.size()-1][3]>0){
                        newpathR= stretching_path(newpathR,contour,true);
                    }
                    if (newpathR[0][3]>0){
                        newpathR= stretching_path(newpathR,contour,false);
                    }
                }
                if (Lstop_cond = false){
                    if (newpathL[newpathL.size()-1][3]>0){
                        newpathL= stretching_path(newpathL,contour,true);
                    }
                    if (newpathL[0][3]>0){
                        newpathL= stretching_path(newpathL,contour,false);
                    }
                }
            } else {
                if (Rstop_cond = false){
                    if (newpathR[newpathR.size()-1][sortdimension]<stretchF){
                        newpathR= stretching_path(newpathR,contour,true);
                    }
                    if (newpathR[0][sortdimension]>stretchB){
                        newpathR= stretching_path(newpathR,contour,false);
                    }
                }
                if (Lstop_cond = false){
                    if (newpathL[newpathL.size()-1][sortdimension]<stretchF){
                        newpathL= stretching_path(newpathL,contour,true);
                    }
                    if (newpathL[0][sortdimension]>stretchB){
                        newpathL= stretching_path(newpathL,contour,false);
                    }
                }
            }
            vector<vector<double>> NnewpathR = subdividePath(newpathR, static_cast<double>(stepsize));
            vector<vector<double>> NnewpathL = subdividePath(newpathL, static_cast<double>(stepsize));
            if (clip_paths_every_iteration == true) {
                //NnewpathR = filterNewPath(NnewpathR, countourpoints);
                NnewpathR = filterInitialPath(NnewpathR, countourpoints);
                NnewpathL = filterInitialPath(NnewpathL, countourpoints);
            }
            std::cout << "found left and right paths" << std::endl;
        }
    }
}


int main() {
    //on linux
    //char buffer[PATH_MAX];
    //if (getcwd(buffer, sizeof(buffer)) != NULL) {
    //    std::string currentPath(buffer);}
    //on windows
    char buffer[MAX_PATH];
    GetModuleFileNameA(NULL, buffer, MAX_PATH);
    std::string fullPath(buffer);
    std::string currentPath = fullPath.substr(0, fullPath.find_last_of("\\/"));
    // comment what is above if you are on linux
    currentPath = deleteLastNCharacters(currentPath,5);
    std::string filename = "Test_Contour.stl";
    std::string filePath = currentPath + "src\\" + filename;  // normally the path is ok but check if it is good

    std::string firstLine;

    bool flipped = false;
    float ZOFFSET = 0.0f; //vertical offset (mm)
    float angleOmit = 30.0f;  // Replace with your desired angle omit value

    if (isASCII(filePath, firstLine)) {
        std::cout << "The file is in ASCII format." << std::endl;
        std::cout << "Reformat the file to binary format and try again." << std::endl;
    }
    else {
        std::cout << "The file is in binary format. Good job, great format" << std::endl;
        std::vector<std::vector<double>> nvecpoints, Nvecs, points;
        readSTL(filePath, angleOmit, points, Nvecs, nvecpoints, ZOFFSET);
        if (flipped) {
            std::cout << "The object will be flipped" << std::endl;
            negateFirstColumn(points);
            negateFirstColumn(nvecpoints);
            negateFirstColumn(Nvecs);
        }
        SLICER_CALCULATIONS(points, Nvecs, nvecpoints);
    }

    return 0;
}
