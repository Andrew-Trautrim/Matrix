#include <cuda_runtime.h>

#include "MatrixKernals.cuh"

namespace MatrixKernals
{
    __global__ void add(double* a, double* b, double* c, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }

        int i = col * m + row;
        c[i] = a[i] + b[i]; 
    }
    
    __global__ void add_broadcast_horizontal(double* a, double* b, double* c, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }

        int i = col * m + row;
        c[i] = a[i] + b[row]; 
    }
    
    __global__ void add_broadcast_vertical(double* a, double* b, double* c, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }

        int i = col * m + row;
        c[i] = a[i] + b[col]; 
    }

    __global__ void subtract(double* a, double* b, double* c, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }

        int i = col * m + row;
        c[i] = a[i] - b[i]; 
    }

    __global__ void subtract_broadcast_horizontal(double* a, double* b, double* c, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }

        int i = col * m + row;
        c[i] = a[i] - b[row]; 
    }

    __global__ void subtract_broadcast_vertical(double* a, double* b, double* c, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }

        int i = col * m + row;
        c[i] = a[i] - b[col]; 
    }
    
    __global__ void multiply(double* a, double* b, double* c, int a_m, int a_n, int b_m, int b_n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= a_m || col >= b_n)
        {
            return;
        }

        double result = 0;
        for (int i = 0; i < a_n; ++i)
        {
            result += a[i * a_m + row] * b[col * b_m + i];
        }

        c[col * a_m + row] = result;
    }

    __global__ void hadamardProduct(double* a, double* b, double* c, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }

        int i = col * m + row;
        c[i] = a[i] * b[i]; 
    }

    __global__ void divide(double* a, double* b, double* c, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }

        int i = col * m + row;
        c[i] = a[i] / b[i]; 
    }

    __global__ void add(double* a, double num, double* b, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }

        int i = col * m + row;
        b[i] = a[i] + num; 
    }

    __global__ void subtract(double* a, double num, double* b, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }

        int i = col * m + row;
        b[i] = a[i] - num; 
    }

    __global__ void multiply(double* a, double num, double* b, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }

        int i = col * m + row;
        b[i] = a[i] * num; 
    }

    __global__ void divide(double* a, double num, double* b, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }

        int i = col * m + row;
        b[i] = a[i] / num; 
    }

    __global__ void transpose(double* a, double* b, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }

        b[row * n + col] = a[col * m + row];
    }

    __global__ void row(double* a, double* b, int row, int m, int n)
    {
        // calculate col for each thread
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (col >= n)
        {
            return;
        }

        b[col] = a[col * m + row];
    }

    __global__ void col(double* a, double* b, int col, int m, int n)
    {
        // calculate row for each thread
        int row = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m)
        {
            return;
        }

        b[row] = a[col * m + row];
    }
    
    __global__ void setup_random_states(curandState* state, unsigned long seed, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }
        
        int i = col * m + row;
        curand_init(seed, i, 0, &state[i]);
    }

    __global__ void randomize_uniform(curandState* state, double* a, int m, int n, int min, int max)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }
        
        int i = col * m + row;

        // get random number between min and max
        curandState localState = state[i];
        double r = (curand_uniform(&localState) * (max - min)) + min;
        state[i] = localState;

        a[i] = r;
    }

    __global__ void randomize_normal(curandState* state, double* a, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }
        
        int i = col * m + row;

        // get random number between min and max
        curandState localState = state[i];
        double r = curand_normal(&localState);
        state[i] = localState;

        a[i] = r;
    }

    __global__ void cross_entropy(double* a, double* b, double* c, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }

        // L(y_hat, y) = y * log(y_hat) + (1 - y) * log(1 - y_hat)
        int i = col * m + row;
        double a_capped = max(min(a[i], 1.0f - 1e-7f), 1e-7f); // make sure a[i] can't be too close to 0 or 1
        c[i] = -1 * (b[i] * log(a_capped) + (1 - b[i]) * log(1 - a_capped));
    }

    __global__ void sigmoid(double* a, double* b, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }

        int i = col * m + row;
        b[i] = 1 / (1 + exp(-1 * a[i])); 
    }

    __global__ void d_sigmoid(double* a, double* b, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }

        int i = col * m + row;
        double s = 1 / (1 + exp(-1 * a[i]));
        b[i] =  s * (1 - s); // d/dx (sigmoid) = sigmoid * (1 - sigmoid) 
    }

    __global__ void tanh(double* a, double* b, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }

        int i = col * m + row;
        b[i] = tanhf(a[i]);
    }

    __global__ void d_tanh(double* a, double* b, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }

        int i = col * m + row;

        // d/dx (tan_h) = 1 - tan_h^2
        double tanh_x = tanhf(a[i]);
        b[i] = 1 - (tanh_x * tanh_x);
    }

    __global__ void relu(double* a, double* b, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }

        int i = col * m + row;
        b[i] = a[i] > 0 ? a[i] : 0;
    }

    __global__ void d_relu(double* a, double* b, int m, int n)
    {
        // Calculate row + col for each thread
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m || col >= n)
        {
            return;
        }

        int i = col * m + row;
        b[i] = a[i] > 0 ? 1 : 0;
    }

    __global__ void softmax(double* a, double* b, int m, int n)
    {
        // Calculate col for each thread
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (col >= n)
        {
            return;
        }

        // get max value of column for stability
        double max_val = -1e308;
        for (int i = 0; i < m; ++i)
        {
            max_val = fmaxf(max_val, a[col * m + i]);
        }
        
        // exponential each value (using max_val for stability)
        double sum = 0;
        for (int i = 0; i < m; ++i)
        {
            b[col * m + i] = exp(a[col * m + i] - max_val);
            sum += b[col * m + i];
        }

        // divide each entry by sum
        for (int i = 0; i < m; ++i)
        {
            b[col * m + i] /= sum;
        }
    }

    __global__ void sum_vertical(double* a, double* b, int m, int n)
    {
        // calculate col for each thread
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (col >= n)
        {
            return;
        }

        double result = 0;
        for (int i = 0; i < m; ++i)
        {
            result += a[col * m + i];
        }

        b[col] = result;
    }

    __global__ void sum_horizontal(double* a, double* b, int m, int n)
    {
        // calculate col for each thread
        int row = blockIdx.x * blockDim.x + threadIdx.x;

        if (row >= m)
        {
            return;
        }

        double result = 0;
        for (int i = 0; i < n; ++i)
        {
            result += a[i * m + row];
        }

        b[row] = result;
    }
}