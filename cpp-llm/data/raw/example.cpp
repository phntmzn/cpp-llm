#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>

// Represents a simple collection of integer values.
class NumberCollection {
private:
    std::vector<int> values_;

public:
    // Adds a value to the collection.
    void add(int value) {
        values_.push_back(value);
    }

    // Returns the sum of all values.
    int sum() const {
        return std::accumulate(
            values_.begin(),
            values_.end(),
            0
        );
    }

    // Returns the arithmetic mean, or 0.0 for an empty collection.
    double average() const {
        if (values_.empty()) {
            return 0.0;
        }

        return static_cast<double>(sum()) /
               static_cast<double>(values_.size());
    }

    // Sorts the values in ascending order.
    void sort() {
        std::sort(values_.begin(), values_.end());
    }

    // Prints all values in the collection.
    void print() const {
        for (std::size_t i = 0; i < values_.size(); ++i) {
            std::cout << values_[i];

            if (i + 1 < values_.size()) {
                std::cout << ", ";
            }
        }

        std::cout << '\n';
    }

    // Returns the number of stored values.
    std::size_t size() const {
        return values_.size();
    }
};

// Returns the larger of two integers.
int maximum(int a, int b) {
    return (a > b) ? a : b;
}

// Checks whether an integer is even.
bool is_even(int value) {
    return value % 2 == 0;
}

int main() {
    NumberCollection numbers;

    numbers.add(42);
    numbers.add(7);
    numbers.add(19);
    numbers.add(84);
    numbers.add(12);

    std::cout << "C++ LLM Training Example\n";
    std::cout << "========================\n";

    std::cout << "Values: ";
    numbers.print();

    std::cout << "Count: " << numbers.size() << '\n';
    std::cout << "Sum: " << numbers.sum() << '\n';
    std::cout << "Average: " << numbers.average() << '\n';

    numbers.sort();

    std::cout << "Sorted values: ";
    numbers.print();

    const int a = 15;
    const int b = 27;

    std::cout << "Maximum of " << a << " and " << b
              << ": " << maximum(a, b) << '\n';

    std::cout << a << (is_even(a) ? " is even\n" : " is odd\n");
    std::cout << b << (is_even(b) ? " is even\n" : " is odd\n");

    return 0;
}e
