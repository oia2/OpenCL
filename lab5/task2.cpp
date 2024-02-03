#include <CL/cl.h>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>


void compareMatrix(const __int64* f, const __int64* s, int size) {
    for (int i = 0; i < size * size; ++i) {
        if (f[i] != s[i]) {
            printf("Matrixes not equal!\n");
            return;
        }
    }
    printf("Matrices are equal!\n");
}

int main()
{
    for (int i = 0; i < 10; ++i) {
        int size = 512;

        size_t byte_size = size * size * sizeof(__int64);
        __int64* Am = (__int64*)malloc(byte_size);
        __int64* Bm = (__int64*)malloc(byte_size);
        __int64* GPU_C = (__int64*)malloc(byte_size);
        __int64* CPU_C = (__int64*)malloc(byte_size);

        for (int i = 0; i < size * size; ++i) {
            Am[i] = rand() % 10;
            Bm[i] = rand() % 10;
            CPU_C[i] = 0;
        }

        printf("Scalar: \n");
        auto start = std::chrono::system_clock::now();
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                for (int k = 0; k < size; ++k) {
                    CPU_C[i * size + j] += Am[i * size + k] * Bm[k * size + j];
                }
            }
        }

        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> diff = end - start;

        printf("Time: %f seconds\n", diff);

        printf("GPU: \n");



        cl_platform_id platform;
        clGetPlatformIDs(1, &platform, NULL);

        cl_device_id device;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

        cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

        cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

        cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, byte_size, Am, NULL);

        cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, byte_size, Bm, NULL);

        cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, byte_size, NULL, NULL);

        start = std::chrono::system_clock::now();

        const char* kernelSource =
            "__kernel void matrixMult(__global const long* Am, __global const long* Bm, __global long* result, int size) {\n"
            "   int gid_x = get_global_id(0);\n"
            "   int gid_y = get_global_id(1);\n"
            "   long sum = 0;\n"
            "   for (int k = 0; k < size; ++k) {\n"
            "       sum += Am[gid_y * size + k] * Bm[k * size + gid_x];\n"
            "   }\n"
            "   result[gid_y * size + gid_x] = sum;\n"
            "}\n";


        cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, NULL);
        cl_int err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

        cl_kernel kernel = clCreateKernel(program, "matrixMult", NULL);

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
        clSetKernelArg(kernel, 3, sizeof(int), &size);

        size_t global_work_size[2] = { size, size };
        clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);

        clFinish(queue);

        end = std::chrono::system_clock::now();
        diff = end - start;
        printf("Time: %f seconds\n", diff);
        clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, byte_size, GPU_C, 0, NULL, NULL);

        compareMatrix(GPU_C, CPU_C, size);

        clReleaseMemObject(d_A);
        clReleaseMemObject(d_B);
        clReleaseMemObject(d_C);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        free(Am);
        free(Bm);
        free(GPU_C);
        free(CPU_C);
    }
    return 0;
}
