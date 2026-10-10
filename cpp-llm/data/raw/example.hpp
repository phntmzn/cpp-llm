#ifndef CPP_LLM_EXAMPLE_HPP
#define CPP_LLM_EXAMPLE_HPP

#include <cstddef>
#include <vector>

// Represents a collection of integer values.
class NumberCollection {
private:
    std::vector<int> values_;

public:
    // Adds a value to the collection.
    void add(int value);

    // Returns the sum of all values.
    int sum() const;

    // Returns the arithmetic mean, or 0.0 if empty.
    double average() const;

    // Sorts the values in ascending order.
    void sort();

    // Prints all values to standard output.
    void print() const;

    // Returns the number of stored values.
    std::size_t size() const;
};

// Returns the larger of two integers.
int maximum(int a, int b);

// Returns true if the integer is even.
bool is_even(int value);

#endif  // CPP_LLM_EXAMPLE_HPP
