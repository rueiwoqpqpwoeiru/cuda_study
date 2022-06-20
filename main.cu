
#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_texture_types.h>
#include <npp.h>
#include <opencv2/opencv.hpp>

class Harris_process {
public:
	Harris_process(const NppiSize image_size) {
		_ker.set_image_size(image_size);
		_buf_in_gray.alloc_2d(image_size);
		_buf_out_gray.alloc_2d(image_size);
		int buffer_size;
		nppiFilterHarrisCornersBorderGetBufferSize(image_size, &buffer_size);
		_buf_forNppFunc.alloc_1d(buffer_size);
	};
	void do_process(const Device_memory<uchar3>& in, Device_memory<uchar3>& out, const float nScale = 1.0e-13, const cudaStream_t& id = 0) {
		_ker.color2gray(in, _buf_in_gray, id);
		nppSetStream(id);
		NppStatus nppStatus = nppiFilterHarrisCornersBorder_8u32f_C1R(_buf_in_gray.ptr(), (int)_buf_in_gray.pitch(), _buf_in_gray.size(), { 0, 0 },
			_buf_out_gray.ptr(), (int)_buf_out_gray.pitch(), _buf_out_gray.size(),
			NPP_FILTER_SOBEL, NPP_MASK_SIZE_5_X_5, NPP_MASK_SIZE_5_X_5, 0.04F, nScale,
			NPP_BORDER_REPLICATE, _buf_forNppFunc.ptr());
		if (NPP_SUCCESS != nppStatus) {
			std::cerr << __FILE__ << "\t" << __LINE__ << "\t" << "NppStatus = " << nppStatus << std::endl;
		}
		_ker.gray2color<Npp32f>(_buf_out_gray, out, id);
	};
private:
	Device_memory<uchar>  _buf_in_gray;
	Device_memory<Npp32f> _buf_out_gray;
	Device_memory<Npp8u>  _buf_forNppFunc;
	Kernel_wrapper _ker;
};

class Host_img_8UC3 {
public:
	void alloc(NppiSize image_size) {
		_size = image_size;
		_data = cv::Mat(_size.height, _size.width, CV_8UC3);
		cudaHostRegister(_data.data, _size.width * _size.height * sizeof(uchar3), cudaHostRegisterDefault);
	};
	cv::Mat& data() { return _data; };
	NppiSize size() const { return _size; };
private:
	cv::Mat _data;
	NppiSize _size;
};

template<typename T> class Device_memory {
public:
	~Device_memory() {
		if (nullptr != _data) {
			cudaFree(_data);
		}
	};
	void alloc_2d(NppiSize image_size) {
		_size = image_size;
		CUDA_CHECK(cudaMallocPitch(&_data, &_pitch, _size.width * sizeof(T), _size.height));
		_host_img_8UC3.alloc(_size);
	};
	void alloc_1d(int array_size) {
		_size = { array_size, 1 };
		_pitch = 0;
		CUDA_CHECK(cudaMalloc(&_data, _size.width * sizeof(T)));
	};
	void imread(const std::string& file_name, const cudaStream_t& id = 0) {
		_host_img_8UC3.data() = cv::imread(file_name);
		_h2d(_host_img_8UC3.data(), id);
	};
	void imwrite(const std::string& file_name, const cudaStream_t& id = 0) {
		_d2h(_host_img_8UC3.data(), id);
		cv::imwrite(file_name, _host_img_8UC3.data());
	}
	T* ptr() const { return _data; };
	NppiSize size() const { return _size; };
	size_t pitch() const { return _pitch; };
	cudaTextureObject_t tex() const { return _tex; };
	// When using texture memory, call the following after alloc_2d
	void use_texture_memory() {
		// Set boundary conditions, subpixel interpolation method, normalize coordinates, etc.
		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.addressMode[0] = cudaAddressModeWrap;
		texDesc.addressMode[1] = cudaAddressModeWrap;
		texDesc.filterMode = cudaFilterModePoint;
		texDesc.normalizedCoords = false;
		texDesc.readMode = cudaReadModeElementType;
		// Set the attributes and parameters of the array to bind
		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.res.pitch2D.desc = cudaCreateChannelDesc<T>();
		resDesc.res.pitch2D.pitchInBytes = _pitch;
		resDesc.res.pitch2D.width = _size.width;
		resDesc.res.pitch2D.height = _size.height;
		resDesc.res.pitch2D.devPtr = _data;
		resDesc.resType = cudaResourceTypePitch2D;
		cudaCreateTextureObject(&_tex, &resDesc, &texDesc, nullptr);
		CUDA_CHECK(cudaGetLastError());
	}
private:
	T* _data;
	NppiSize _size;
	size_t _pitch;
	cudaTextureObject_t _tex;
	Host_img_8UC3 _host_img_8UC3;
	void _h2d(const cv::Mat& in, const cudaStream_t& id = 0) {
		if (0 != _pitch) {
			CUDA_CHECK(cudaMemcpy2DAsync(_data, _pitch, in.data, in.step, _size.width * sizeof(T), _size.height, cudaMemcpyHostToDevice, id));
		}
		else {
			CUDA_CHECK(cudaMemcpyAsync(_data, in.data, _size.width * sizeof(T), cudaMemcpyHostToDevice, id));
		}
	};
	void _d2h(cv::Mat& out, const cudaStream_t& id = 0) {
		if (0 != _pitch) {
			CUDA_CHECK(cudaMemcpy2DAsync(out.data, out.step, _data, _pitch, _size.width * sizeof(T), _size.height, cudaMemcpyDeviceToHost, id));
		}
		else {
			CUDA_CHECK(cudaMemcpyAsync(out.data, _data, _size.width * sizeof(T), cudaMemcpyDeviceToHost, id));
		}
	};
};

