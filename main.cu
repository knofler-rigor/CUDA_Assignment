#include <stdio.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/random.h>

void prefixSum(const thrust::device_vector<float> &vector)
{
    thrust::device_vector<float> outVector(vector.size());

    std::cout << "\nPrefix Sum: Inclusive scan:" << std::endl;
    thrust::inclusive_scan(vector.begin(), vector.end(), outVector.begin());
    printData(outVector);

    std::cout << "\nPrefix Sum: Exclusive scan:" << std::endl;
    thrust::exclusive_scan(vector.begin(), vector.end(), outVector.begin());
    printData(outVector);
}

struct is_negative
{
    __host__ __device__ bool operator()(const int x)
    {
        return x < 0.0;
    }
};

struct saxpy_functor
{
    const float a;

    saxpy_functor(float _a) : a(_a) {}

    __host__ __device__ float operator()(const float &x, const float &y) const
    {
        return a * x + y;
    }
};

void reorder(const thrust::device_vector<float> &vector)
{
    thrust::device_vector<float> outVector(vector.size());

    std::cout << "\nReordering: Unique values:" << std::endl;
    const auto newUniqueItEnd = thrust::unique_copy(vector.cbegin(), vector.cend(), outVector.begin());
    printData(outVector);

    std::cout << "\nReordering: Remove negative values:" << std::endl;
    const auto newRemoveItEnd = thrust::remove_copy_if(vector.cbegin(), vector.cend(), outVector.begin(), is_negative());
    printData(outVector);
}

void sort(const thrust::device_vector<float> &vector)
{
    thrust::device_vector<float> outVector = vector;

    std::cout << "\nSorting: Unstable sort:" << std::endl;
    thrust::sort(outVector.begin(), outVector.end());
    printData(outVector);

    outVector = vector;

    std::cout << "\nSorting: Stable sort:" << std::endl;
    thrust::stable_sort(outVector.begin(), outVector.end());
    printData(outVector);
}

void transform(const thrust::device_vector<float> &vector)
{
    thrust::device_vector<float> outVector(vector.size());

    std::cout << "\nTransformation: Reverse sign:" << std::endl;
    thrust::transform(vector.cbegin(), vector.cend(), outVector.begin(), thrust::negate<float>());
    printData(outVector);

    std::cout << "\nTransformation: SAXPY transform:" << std::endl;
    thrust::transform(vector.cbegin(), vector.cend(), vector.begin(), outVector.begin(), saxpy_functor(1.0));
    printData(outVector);

    std::cout << "\nTransformation: Vector Multiplication:" << std::endl;
    thrust::device_vector<float> base(vector.size());
    thrust::sequence(base.begin(), base.end());
    thrust::transform(vector.cbegin(), vector.cend(), base.begin(), outVector.begin(), thrust::multiplies<float>());
    printData(outVector);
}

void printData(const thrust::device_vector<float> &vector)
{
    for (int i = 0; i < vector.size(); ++i)
    {
        std::cout << vector[i];
        if (i < vector.size() - 1)
        {
            std::cout << " ";
        }
        else
        {
            std::cout << std::endl;
        }
    }
}

int main(int argc, char *argv[])
{
    printf("%s Starting...\n\n", argv[0]);

    int size = 0;
    if (argc >= 2)
    {
        char *inputData = argv[1];
        if (!inputData)
        {
            std::cerr << "Cannot read the dataset size!" << std::endl;
            exit(EXIT_FAILURE);
        }
        size = std::atoi(inputData);
    }
    else
    {
        std::cerr << "Dataset size missed!" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string algorithm;
    if (argc >= 3)
    {
        char *algorithmData = argv[2];
        if (!algorithmData)
        {
            std::cerr << "Cannot read the algorithm type!" << std::endl;
            exit(EXIT_FAILURE);
        }
        algorithm = std::string(algorithmData);
    }
    else
    {
        std::cerr << "Algorithm type is missed!" << std::endl;
        exit(EXIT_FAILURE);
    }

    thrust::device_vector<float> data(size);

    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<float> dist(-100, 100);
    for (int i = 0; i < size; ++i)
    {
        data[i] = dist(rng);
    }
    std::cout << "Input vector: " << std::endl;
    printData(data);

    if (algorithm == "transformation" || algorithm == "transform")
    {
        std::cout << "\nSelected Transformation Algorithms:" << std::endl;
        transform(data);
    }
    else if (algorithm == "reduction")
    {
        std::cout << "\nSelected Reduction Algorithhms:" << std::endl;
        reduce(data);
    }
    else if (algorithm == "prefix-sum")
    {
        std::cout << "\nSelected Prefix-sum Algorithms:" << std::endl;
        prefixSum(data);
    }
    else if (algorithm == "reordering")
    {
        std::cout << "\nSelected Reordering Algorithms:" << std::endl;
        reorder(data);
    }
    else if (algorithm == "sorting")
    {
        std::cout << "\nSelected Sorting Algorithms:" << std::endl;
        sort(data);
    }
    else
    {
        std::cout << "Algorithm type isn't supported!" << std::endl;
    }

    return 0;
}

void reduce(const thrust::device_vector<float> &vector)
{
    std::cout << "\nReduction: Reduce sum:" << std::endl;
    float sum = thrust::reduce(vector.begin(), vector.end(), 0.0, thrust::plus<float>());
    std::cout << sum << std::endl;

    std::cout << "\nReduction: Min element:" << std::endl;
    const auto min = thrust::min_element(vector.cbegin(), vector.cend());
    std::cout << (*min) << std::endl;

    std::cout << "\nReduction: Max element:" << std::endl;
    const auto max = thrust::max_element(vector.cbegin(), vector.cend());
    std::cout << (*max) << std::endl;
}
