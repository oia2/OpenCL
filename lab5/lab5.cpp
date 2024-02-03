#include <iostream>
#include <opencv2/opencv.hpp>
#include <CL/cl.h>
#include <chrono>


void computeChannelsGPU(const cv::Mat& image, cv::Mat& modifiedBlueChannel, cv::Mat& modifiedYellowChannel) {
    // Получение платформы
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    // Получение устройства
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    // Создание контекста
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

    // Создание очереди
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

    const char* source = R"(
        __kernel void computeChannels(__global const uchar* img,
                                      __global uchar* blueChannel,
                                      __global uchar* yellowChannel,
                                      const int rows,
                                      const int cols) {
            int x = get_global_id(0);
            int y = get_global_id(1);

            if (x < cols && y < rows) {
                int offset = y * cols + x;

                uchar redv = img[offset * 3 + 2];
                uchar greenv = img[offset * 3 + 1];
                uchar bluev = img[offset * 3];

                // Вычисление модифицированного синего канала
                uchar Bv = bluev - (greenv + bluev) / 2;

                // Вычисление модифицированного желтого канала
                uchar Yv = redv + greenv - 2 * (abs(redv - greenv) + bluev);

                // Запись результатов
                blueChannel[offset] = Bv;
                yellowChannel[offset] = Yv;
            }
        }
    )";

    cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, NULL);

    // Компиляция программы
    cl_int err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    // Создание ядра
    cl_kernel kernel = clCreateKernel(program, "computeChannels", NULL);

    // Выделение буферов памяти на устройстве
    cl_mem d_img = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uchar) * image.rows * image.cols * 3, (void*)image.data, NULL);
    cl_mem d_blueChannel = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        sizeof(uchar) * image.rows * image.cols, NULL, NULL);
    cl_mem d_yellowChannel = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        sizeof(uchar) * image.rows * image.cols, NULL, NULL);

    // Установка параметров ядра
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_img);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_blueChannel);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_yellowChannel);
    clSetKernelArg(kernel, 3, sizeof(int), &image.rows);
    clSetKernelArg(kernel, 4, sizeof(int), &image.cols);

    size_t global_work_size[2] = { static_cast<size_t>(image.cols), static_cast<size_t>(image.rows) };


    auto start = std::chrono::high_resolution_clock::now();

    // Запуск ядра на устройстве
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
    clFinish(queue);

    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "GPU Time " << diff.count() << " ms" << std::endl;

    // Копирование результатов обратно на хост
    clEnqueueReadBuffer(queue, d_blueChannel, CL_TRUE, 0, sizeof(uchar) * image.rows * image.cols, modifiedBlueChannel.data, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, d_yellowChannel, CL_TRUE, 0, sizeof(uchar) * image.rows * image.cols, modifiedYellowChannel.data, 0, NULL, NULL);

    // Освобождение ресурсов OpenCL
    clReleaseMemObject(d_img);
    clReleaseMemObject(d_blueChannel);
    clReleaseMemObject(d_yellowChannel);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

int main() {
    for (int i = 1; i <= 10; i++) {
        cv::Mat image = cv::imread("../src/giena_1024x768.jpg");

        if (image.empty()) {
            std::cout << "Failed to upload image." << std::endl;
            return -1;
        }

        // Создание двух пустых Mat-объектов для модифицированных синего и желтого каналов
        cv::Mat modifiedBlueChannel = cv::Mat::zeros(image.size(), CV_8U);
        cv::Mat modifiedYellowChannel = cv::Mat::zeros(image.size(), CV_8U);

        // Вызов функции для вычисления каналов на GPU
        computeChannelsGPU(image, modifiedBlueChannel, modifiedYellowChannel);

        // Сохранение изображений в файлы
        cv::imwrite("../modified_blue_channel_gpu.jpg", modifiedBlueChannel);
        cv::imwrite("../modified_yellow_channel_gpu.jpg", modifiedYellowChannel);
    }

    return 0;
}
1024x768
