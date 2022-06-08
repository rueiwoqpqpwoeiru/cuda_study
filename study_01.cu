
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <npp.h>

template<typename T> struct Data_for_kernel {
	T* ptr;
	NppiSize size;
	size_t pitch;
	__device__ __forceinline__ T& val(unsigned int x, unsigned int y) const { return ((T*)((char*)ptr + pitch * y))[x]; }
};

template<typename T> class Device_memory {
public:
	Device_memory(NppiSize image_size) {
		_d.size = image_size;
		cudaMallocPitch(&_d.ptr, &_d.pitch, _d.size.width * sizeof(T), _d.size.height);
	}
	~Device_memory() {
		if (nullptr != _d.ptr) {
			cudaFree(_d.ptr);
		}
	};
	void imread(const std::string& file_name) {
		cv::Mat in = cv::imread(file_name);
		cudaMemcpy2D(_d.ptr, _d.pitch, in.data, in.step, _d.size.width * sizeof(T), _d.size.height, cudaMemcpyHostToDevice);
	};
	void imwrite(const std::string& file_name) {
		cv::Mat out = cv::Mat(_d.size.height, _d.size.width, CV_8UC3);
		cudaMemcpy2D(out.data, out.step, _d.ptr, _d.pitch, _d.size.width * sizeof(T), _d.size.height, cudaMemcpyDeviceToHost);
		cv::imwrite(file_name, out);
	}
	Data_for_kernel<T> data_for_kernel() const { return _d; }
private:
	Data_for_kernel<T> _d;
};

class Process_size {
public:
	Process_size(const NppiSize& image_size, const dim3& block) {
		set(image_size, block);
	}
	void set(const NppiSize& image_size, const dim3& block) {
		_block = block;
		_grid = dim3((image_size.width + block.x - 1) / block.x, (image_size.height + block.y - 1) / block.y);
	};
	dim3 block() const { return _block; }
	dim3 grid()  const { return _grid; }
private:
	dim3 _block;
	dim3 _grid;
};

namespace Kernel_func {
	__global__ void kernel(const Data_for_kernel<uchar3> in, Data_for_kernel<uchar3> out) {
		unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
		unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
		if (x < in.size.width && y < in.size.height) {
			uchar3 val_in = in.val(x, y);
			int val_out = (val_in.x + 2 * val_in.y + val_in.z) / 4;
			out.val(x, y).x = static_cast<uchar>(val_out);
			out.val(x, y).y = static_cast<uchar>(val_out);
			out.val(x, y).z = static_cast<uchar>(val_out);
		}
	}
};

int main(int argc, char* argv[]) {
	// 1枚目の画像からサイズを取得
	cv::Mat tmp = cv::imread("in_0.tif");
	NppiSize image_size = {tmp.size().width, tmp.size().height};
	// デバイスメモリを確保
	Device_memory<uchar3> in_img(image_size);
	Device_memory<uchar3> out_img(image_size);
	// GPUでの処理サイズ
	Process_size process_size(image_size, dim3(32, 8));
	// ループ
	for (unsigned int i = 0; i < 1; i++) {
		in_img.imread("in_" + std::to_string(i) + ".tif");
		Kernel_func::kernel << < process_size.grid(), process_size.block() >> > (in_img.data_for_kernel(), out_img.data_for_kernel());
		out_img.imwrite("out_" + std::to_string(i) + ".tif");
	}
	cudaDeviceReset();
	return 0;
}

