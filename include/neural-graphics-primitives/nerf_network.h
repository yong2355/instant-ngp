/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   nerf_network.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  A network that first processes 3D position to density and
 *          subsequently direction to color.
 */

#pragma once

#include <tiny-cuda-nn/common.h>

#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/multi_stream.h>
#include <tiny-cuda-nn/network.h>
#include <iostream>
#include <tiny-cuda-nn/network_with_input_encoding.h>

NGP_NAMESPACE_BEGIN

template <typename T>
__global__ void extract_density(
	const uint32_t n_elements,
	const uint32_t density_stride,
	const uint32_t rgbd_stride,
	const T* __restrict__ density,
	T* __restrict__ rgbd
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	rgbd[i * rgbd_stride] = density[i * density_stride];
}

template <typename T>
__global__ void extract_rgb(
	const uint32_t n_elements,
	const uint32_t rgb_stride,
	const uint32_t output_stride,
	const T* __restrict__ rgbd,
	T* __restrict__ rgb
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i / 3;
	const uint32_t dim_idx = i - elem_idx * 3;

	rgb[elem_idx*rgb_stride + dim_idx] = rgbd[elem_idx*output_stride + dim_idx];
}

template <typename T>
__global__ void add_density_gradient(
	const uint32_t n_elements,
	const uint32_t rgbd_stride,
	const T* __restrict__ rgbd,
	const uint32_t density_stride,
	T* __restrict__ density
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	density[i * density_stride] += rgbd[i * rgbd_stride + 3];
}

template <typename T>
__device__ float myownmax(float* address, float val)
{
	int* address_as_int = (int*)address;
	int old = *address_as_int, assumed;
	do{
		assumed = old;
		old = atomicCAS(address_as_int, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
	}while(assumed != old);
	return __int_as_float(old);
}
template <typename T>
__global__ void get_scale(
	const uint32_t num_elements,
	T* __restrict__ encoded_positions,
	float* scale_factor
) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i==0){printf("start get_scale!!!!\n");}
	if(i>=32*num_elements)return;
	float max1 = (float)encoded_positions[0];
	float min1 = (float)encoded_positions[0];
	while(i<32*num_elements){
		if((float)encoded_positions[i]>max1){max1 = (float)encoded_positions[i];}
		if((float)encoded_positions[i]<min1){min1 = (float)encoded_positions[i];}
		i += blockDim.x*gridDim.x;
	}
	float scale = (max1>-min1) ? 2*max1 : -2*min1;
	myownmax<T>(scale_factor, scale);
}
template <typename T>
__global__ void quantize(
	const uint32_t num_elements,
	T* __restrict__ encoded_positions,
	float* scale_factor
) {
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i>=num_elements*32)return;
	int max_int = 127;
	int min_int = -128;
	int range = 255;
	if(i==0){printf("start quantize!!!!\n");}
	
	float x = floorf(((float)encoded_positions[i])/(*scale_factor)*range + 0.5f);
	encoded_positions[i] = (T)((x >= max_int) ? max_int : (x <= min_int ? min_int : x));
	//encoded_positions[i] = (T)((x >= max_int) ? max_int*(*scale_factor)/range : (x <= min_int ? min_int*(*scale_factor)/range : x*(*scale_factor)/range));
}
template <typename T>
__global__ void dequantize(
	const uint32_t num_elements,
	T* __restrict__ encoded_positions,
	float* scale_factor,
	float mlp_scale
) {
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i>=num_elements)return;
	int range = 255;
	if(i==0){printf("start dequantize!!!!\n");}
	
	//encoded_positions[i] = (T)((float)encoded_positions[i]*mlp_scale*(*scale_factor/range));
	encoded_positions[i] = (T)((float)encoded_positions[i]*(*scale_factor/range));
}
template <typename T>
__global__ void truncate_overflow(
	const uint32_t num_elements,
	T* __restrict__ encoded_positions,
	float* scale_factor,
	float mlp_scale
) {
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i>=num_elements)return;
	int range = 255;
	int max_int = 127;
	int min_int = -128;
	if(i==0){printf("start truncate!!!!\n");}
	if((float)encoded_positions[i]>=max_int*(*scale_factor)*mlp_scale/range){encoded_positions[i] = max_int*(*scale_factor)*mlp_scale/range;}
	if((float)encoded_positions[i]<=min_int*(*scale_factor)*mlp_scale/range){encoded_positions[i] = min_int*(*scale_factor)*mlp_scale/range;}
}

template <typename T>
class NerfNetwork : public tcnn::Network<float, T> {
public:
	using json = nlohmann::json;

	NerfNetwork(uint32_t n_pos_dims, uint32_t n_dir_dims, uint32_t n_extra_dims, uint32_t dir_offset, const json& pos_encoding, const json& dir_encoding, const json& density_network, const json& rgb_network) : m_n_pos_dims{n_pos_dims}, m_n_dir_dims{n_dir_dims}, m_dir_offset{dir_offset}, m_n_extra_dims{n_extra_dims} {
		m_pos_encoding.reset(tcnn::create_encoding<T>(n_pos_dims, pos_encoding, density_network.contains("otype") && (tcnn::equals_case_insensitive(density_network["otype"], "FullyFusedMLP") || tcnn::equals_case_insensitive(density_network["otype"], "MegakernelMLP")) ? 16u : 8u));
		uint32_t rgb_alignment = tcnn::minimum_alignment(rgb_network);
		m_dir_encoding.reset(tcnn::create_encoding<T>(m_n_dir_dims + m_n_extra_dims, dir_encoding, rgb_alignment));

		json local_density_network_config = density_network;
		local_density_network_config["n_input_dims"] = m_pos_encoding->padded_output_width();
		if (!density_network.contains("n_output_dims")) {
			local_density_network_config["n_output_dims"] = 16;
		}
		m_density_network.reset(tcnn::create_network<T>(local_density_network_config));

		m_rgb_network_input_width = tcnn::next_multiple(m_dir_encoding->padded_output_width() + m_density_network->padded_output_width(), rgb_alignment);

		json local_rgb_network_config = rgb_network;
		local_rgb_network_config["n_input_dims"] = m_rgb_network_input_width;
		local_rgb_network_config["n_output_dims"] = 3;
		m_rgb_network.reset(tcnn::create_network<T>(local_rgb_network_config));
	}

	virtual ~NerfNetwork() { }

	void inference_mixed_precision_impl(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>& output, bool use_inference_params = true) override {
		std::cout << "we executed inference_mixed_precision_impl of nerf_network.h\n";
		uint32_t batch_size = input.n();
		tcnn::GPUMatrixDynamic<T> density_network_input{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		tcnn::GPUMatrixDynamic<T> rgb_network_input{m_rgb_network_input_width, batch_size, stream, m_dir_encoding->preferred_output_layout()};

		tcnn::GPUMatrixDynamic<T> density_network_output = rgb_network_input.slice_rows(0, m_density_network->padded_output_width());
		tcnn::GPUMatrixDynamic<T> rgb_network_output{output.data(), m_rgb_network->padded_output_width(), batch_size, output.layout()};
		std::cout << "line before m_pos_encoding->inference_mixed_precision in inference_mixed_precision_impl fn of nerf_network.h\n";
		m_pos_encoding->inference_mixed_precision(
			stream,
			input.slice_rows(0, m_pos_encoding->input_width()),
			density_network_input,
			use_inference_params
		);
		std::cout << "line after m_pos_encoding->inference_mixed_precision in inference_mixed_precision_impl fn of nerf_network.h\n";

		/*std::cout << "Start Quantize_1!!\n";
		cudaDeviceSynchronize();
		T* target1 = density_network_input.data();
		float* scale_factor1;
		cudaMalloc(&scale_factor1, 4);
		float float_min = -10000;
		cudaMemcpy(scale_factor1,&float_min,4,cudaMemcpyHostToDevice);
		uint32_t dg = (32*batch_size + 1024 - 1) / 1024;
		get_scale<T><<<dg, 1024>>>(batch_size, target1, scale_factor1);
		cudaDeviceSynchronize();
		const dim3 tt = { dg, 1, 1 };
		quantize<T><<<tt, 1024, 0>>>(batch_size, target1, scale_factor1);
		cudaDeviceSynchronize();
		//cudaFree(scale_factor1);*/

		std::cout << "line before m_density_network->inference_mixed_precision in inference_mixed_precision_impl fn of nerf_network.h\n";
		m_density_network->inference_mixed_precision(stream, density_network_input, density_network_output, use_inference_params);
		cudaDeviceSynchronize();
		std::cout << "line after m_density_network->inference_mixed_precision in inference_mixed_precision_impl fn of nerf_network.h\n";

		auto dir_out = rgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width());

		std::cout << "line before m_dir_encoding->inference_mixed_precision in inference_mixed_precision_impl fn of nerf_network.h\n";
		m_dir_encoding->inference_mixed_precision(
			stream,
			input.slice_rows(m_dir_offset, m_dir_encoding->input_width()),
			dir_out,
			use_inference_params
		);
		std::cout << "line after m_dir_encoding->inference_mixed_precision in inference_mixed_precision_impl fn of nerf_network.h\n";

		cudaDeviceSynchronize();
		T* temp=(T*)malloc(batch_size*sizeof(__half)*16);
		cudaMemcpy(temp,dir_out.data(),batch_size*32,cudaMemcpyDeviceToHost);
		float max = (float)temp[0];
		float min = (float)temp[0];
		float sum = (float)temp[0];
		for(int i=1;i<16*batch_size;i++){
			if(max<(float)temp[i]){max=(float)temp[i];}
			if(min>(float)temp[i]){min=(float)temp[i];}
			sum = sum+(float)temp[i];
		}
		float interval = (float)(max-min)/10;
		int histo[10] = {0,};
		for(int i=0;i<16*batch_size;i++){
			if((float)temp[i]<min + interval){histo[0]++;}
			else if((float)temp[i]<min + interval*2){histo[1]++;}
			else if((float)temp[i]<min + interval*3){histo[2]++;}
			else if((float)temp[i]<min + interval*4){histo[3]++;}
			else if((float)temp[i]<min + interval*5){histo[4]++;}
			else if((float)temp[i]<min + interval*6){histo[5]++;}
			else if((float)temp[i]<min + interval*7){histo[6]++;}
			else if((float)temp[i]<min + interval*8){histo[7]++;}
			else if((float)temp[i]<min + interval*9){histo[8]++;}
			else {histo[9]++;}
		}
		printf("max is %f, min is %f, mean is %f\n",max,min,(sum/(16*batch_size)));
		printf("histo is %d, %d, %d, %d, %d, %d, %d, %d, %d, %d\n",histo[0],histo[1],histo[2],histo[3],histo[4],histo[5],histo[6],histo[7],histo[8],histo[9]);
		
		free(temp);

		/*float density_network_scales = 0.004486083984375 * 0.006870269775390625;	//int8
		//float density_network_scales = 0.035888671875*0.054962158203125;	//test(*2.5)
		//float density_network_scales = 0.004833221435546875 * 0.00616455078125;	//31k int8
		//float density_network_scales = 0.07623291015625 * 0.1168212890625;			//int4
		uint32_t dg1 = (16*batch_size + 1024 - 1) / 1024;
		const dim3 tt1 = { dg1, 1, 1 };
		T* target3 = density_network_output.data();
		dequantize<T><<<tt1,1024,0>>>(16*batch_size,target3, scale_factor1,density_network_scales);
		cudaDeviceSynchronize();
		//truncate_overflow<T><<<tt1, 1024, 0>>>(16*batch_size, target3, scale_factor1, density_network_scales);
		//cudaDeviceSynchronize();
		cudaFree(scale_factor1);

		std::cout << "Start Quantize_2!!\n";
		T* target2 = rgb_network_input.data();
		float* scale_factor2;
		cudaMalloc(&scale_factor2, 4);
		cudaMemcpy(scale_factor2,&float_min,4,cudaMemcpyHostToDevice);
		//uint32_t dg = (m_rgb_network_input_width*batch_size + 1024 - 1) / 1024;
		get_scale<T><<<dg, 1024>>>(batch_size, target2, scale_factor2);
		cudaDeviceSynchronize();
		//const dim3 tt = { dg, 1, 1 };
		quantize<T><<<tt, 1024, 0>>>(batch_size, target2, scale_factor2);
		cudaDeviceSynchronize();
		//cudaFree(scale_factor2);*/

		std::cout << "line before m_rgb_network->inference_mixed_precision in inference_mixed_precision_impl fn of nerf_network.h\n";
		m_rgb_network->inference_mixed_precision(stream, rgb_network_input, rgb_network_output, use_inference_params);
		std::cout << "line after m_rgb_network->inference_mixed_precision in inference_mixed_precision_impl fn of nerf_network.h\n";
		
		/*float rgb_network_scales = 0.007289886474609375 * 0.00589752197265625 * 0.0080413818359375;	//int8
		//float rgb_network_scales = 0.058319091796875*0.04718017578125*0.0643310546875;	//test
		//float rgb_network_scales = 0.00579071044921875 * 0.00600433349609375 * 0.00770568847656250;	//31k int8
		//float rgb_network_scales = 0.12396240234375 * 0.10028076171875 * 0.13671875;				//int4
		uint32_t dg2 = (3*batch_size + 1024 - 1) / 1024;
		const dim3 tt2 = { dg2, 1, 1 };
		dequantize<T><<<tt2,1024,0>>>(3*batch_size,rgb_network_output.data(),scale_factor2,rgb_network_scales);
		cudaDeviceSynchronize();
		//truncate_overflow<T><<<tt2, 1024, 0>>>(3*batch_size, rgb_network_output.data(), scale_factor2, rgb_network_scales);
		//cudaDeviceSynchronize();
		cudaFree(scale_factor2);*/
		tcnn::linear_kernel(extract_density<T>, 0, stream,	//density, rgbd stride are 1
			batch_size,
			density_network_output.layout() == tcnn::AoS ? density_network_output.stride() : 1,
			output.layout() == tcnn::AoS ? padded_output_width() : 1,
			density_network_output.data(),
			output.data() + 3 * (output.layout() == tcnn::AoS ? 1 : batch_size)
		);
	}

	uint32_t padded_density_output_width() const {
		return m_density_network->padded_output_width();
	}

	std::unique_ptr<tcnn::Context> forward_impl(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>* output = nullptr, bool use_inference_params = false, bool prepare_input_gradients = false) override {
		// Make sure our temporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();

		auto forward = std::make_unique<ForwardContext>();

		forward->density_network_input = tcnn::GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		forward->rgb_network_input = tcnn::GPUMatrixDynamic<T>{m_rgb_network_input_width, batch_size, stream, m_dir_encoding->preferred_output_layout()};

		forward->pos_encoding_ctx = m_pos_encoding->forward(
			stream,
			input.slice_rows(0, m_pos_encoding->input_width()),
			&forward->density_network_input,
			use_inference_params,
			prepare_input_gradients
		);

		forward->density_network_output = forward->rgb_network_input.slice_rows(0, m_density_network->padded_output_width());
		forward->density_network_ctx = m_density_network->forward(stream, forward->density_network_input, &forward->density_network_output, use_inference_params, prepare_input_gradients);

		auto dir_out = forward->rgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width());
		forward->dir_encoding_ctx = m_dir_encoding->forward(
			stream,
			input.slice_rows(m_dir_offset, m_dir_encoding->input_width()),
			&dir_out,
			use_inference_params,
			prepare_input_gradients
		);

		if (output) {
			forward->rgb_network_output = tcnn::GPUMatrixDynamic<T>{output->data(), m_rgb_network->padded_output_width(), batch_size, output->layout()};
		}

		forward->rgb_network_ctx = m_rgb_network->forward(stream, forward->rgb_network_input, output ? &forward->rgb_network_output : nullptr, use_inference_params, prepare_input_gradients);

		if (output) {
			tcnn::linear_kernel(extract_density<T>, 0, stream,
				batch_size, m_dir_encoding->preferred_output_layout() == tcnn::AoS ? forward->density_network_output.stride() : 1, padded_output_width(), forward->density_network_output.data(), output->data()+3
			);
		}

		return forward;
	}

	void backward_impl(
		cudaStream_t stream,
		const tcnn::Context& ctx,
		const tcnn::GPUMatrixDynamic<float>& input,
		const tcnn::GPUMatrixDynamic<T>& output,
		const tcnn::GPUMatrixDynamic<T>& dL_doutput,
		tcnn::GPUMatrixDynamic<float>* dL_dinput = nullptr,
		bool use_inference_params = false,
		tcnn::EGradientMode param_gradients_mode = tcnn::EGradientMode::Overwrite
	) override {
		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);

		// Make sure our teporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();

		tcnn::GPUMatrix<T> dL_drgb{m_rgb_network->padded_output_width(), batch_size, stream};
		CUDA_CHECK_THROW(cudaMemsetAsync(dL_drgb.data(), 0, dL_drgb.n_bytes(), stream));
		tcnn::linear_kernel(extract_rgb<T>, 0, stream,
			batch_size*3, dL_drgb.m(), dL_doutput.m(), dL_doutput.data(), dL_drgb.data()
		);

		const tcnn::GPUMatrixDynamic<T> rgb_network_output{(T*)output.data(), m_rgb_network->padded_output_width(), batch_size, output.layout()};
		tcnn::GPUMatrixDynamic<T> dL_drgb_network_input{m_rgb_network_input_width, batch_size, stream, m_dir_encoding->preferred_output_layout()};
		m_rgb_network->backward(stream, *forward.rgb_network_ctx, forward.rgb_network_input, rgb_network_output, dL_drgb, &dL_drgb_network_input, use_inference_params, param_gradients_mode);

		// Backprop through dir encoding if it is trainable or if we need input gradients
		if (m_dir_encoding->n_params() > 0 || dL_dinput) {
			tcnn::GPUMatrixDynamic<T> dL_ddir_encoding_output = dL_drgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width());
			tcnn::GPUMatrixDynamic<float> dL_ddir_encoding_input;
			if (dL_dinput) {
				dL_ddir_encoding_input = dL_dinput->slice_rows(m_dir_offset, m_dir_encoding->input_width());
			}

			m_dir_encoding->backward(
				stream,
				*forward.dir_encoding_ctx,
				input.slice_rows(m_dir_offset, m_dir_encoding->input_width()),
				forward.rgb_network_input.slice_rows(m_density_network->padded_output_width(), m_dir_encoding->padded_output_width()),
				dL_ddir_encoding_output,
				dL_dinput ? &dL_ddir_encoding_input : nullptr,
				use_inference_params,
				param_gradients_mode
			);
		}

		tcnn::GPUMatrixDynamic<T> dL_ddensity_network_output = dL_drgb_network_input.slice_rows(0, m_density_network->padded_output_width());
		tcnn::linear_kernel(add_density_gradient<T>, 0, stream,
			batch_size,
			dL_doutput.m(),
			dL_doutput.data(),
			dL_ddensity_network_output.layout() == tcnn::RM ? 1 : dL_ddensity_network_output.stride(),
			dL_ddensity_network_output.data()
		);

		tcnn::GPUMatrixDynamic<T> dL_ddensity_network_input;
		if (m_pos_encoding->n_params() > 0 || dL_dinput) {
			dL_ddensity_network_input = tcnn::GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		}

		m_density_network->backward(stream, *forward.density_network_ctx, forward.density_network_input, forward.density_network_output, dL_ddensity_network_output, dL_ddensity_network_input.data() ? &dL_ddensity_network_input : nullptr, use_inference_params, param_gradients_mode);

		// Backprop through pos encoding if it is trainable or if we need input gradients
		if (dL_ddensity_network_input.data()) {
			tcnn::GPUMatrixDynamic<float> dL_dpos_encoding_input;
			if (dL_dinput) {
				dL_dpos_encoding_input = dL_dinput->slice_rows(0, m_pos_encoding->input_width());
			}

			m_pos_encoding->backward(
				stream,
				*forward.pos_encoding_ctx,
				input.slice_rows(0, m_pos_encoding->input_width()),
				forward.density_network_input,
				dL_ddensity_network_input,
				dL_dinput ? &dL_dpos_encoding_input : nullptr,
				use_inference_params,
				param_gradients_mode
			);
		}
	}

	void density(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>& output, bool use_inference_params = true) {
		if (input.layout() != tcnn::CM) {
			throw std::runtime_error("NerfNetwork::density input must be in column major format.");
		}

		uint32_t batch_size = output.n();
		tcnn::GPUMatrixDynamic<T> density_network_input{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};

		m_pos_encoding->inference_mixed_precision(
			stream,
			input.slice_rows(0, m_pos_encoding->input_width()),
			density_network_input,
			use_inference_params
		);

		m_density_network->inference_mixed_precision(stream, density_network_input, output, use_inference_params);
	}

	std::unique_ptr<tcnn::Context> density_forward(cudaStream_t stream, const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<T>* output = nullptr, bool use_inference_params = false, bool prepare_input_gradients = false) {
		if (input.layout() != tcnn::CM) {
			throw std::runtime_error("NerfNetwork::density_forward input must be in column major format.");
		}

		// Make sure our temporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();

		auto forward = std::make_unique<ForwardContext>();

		forward->density_network_input = tcnn::GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};

		forward->pos_encoding_ctx = m_pos_encoding->forward(
			stream,
			input.slice_rows(0, m_pos_encoding->input_width()),
			&forward->density_network_input,
			use_inference_params,
			prepare_input_gradients
		);

		if (output) {
			forward->density_network_output = tcnn::GPUMatrixDynamic<T>{output->data(), m_density_network->padded_output_width(), batch_size, output->layout()};
		}

		forward->density_network_ctx = m_density_network->forward(stream, forward->density_network_input, output ? &forward->density_network_output : nullptr, use_inference_params, prepare_input_gradients);

		return forward;
	}

	void density_backward(
		cudaStream_t stream,
		const tcnn::Context& ctx,
		const tcnn::GPUMatrixDynamic<float>& input,
		const tcnn::GPUMatrixDynamic<T>& output,
		const tcnn::GPUMatrixDynamic<T>& dL_doutput,
		tcnn::GPUMatrixDynamic<float>* dL_dinput = nullptr,
		bool use_inference_params = false,
		tcnn::EGradientMode param_gradients_mode = tcnn::EGradientMode::Overwrite
	) {
		if (input.layout() != tcnn::CM || (dL_dinput && dL_dinput->layout() != tcnn::CM)) {
			throw std::runtime_error("NerfNetwork::density_backward input must be in column major format.");
		}

		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);

		// Make sure our temporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();

		tcnn::GPUMatrixDynamic<T> dL_ddensity_network_input;
		if (m_pos_encoding->n_params() > 0 || dL_dinput) {
			dL_ddensity_network_input = tcnn::GPUMatrixDynamic<T>{m_pos_encoding->padded_output_width(), batch_size, stream, m_pos_encoding->preferred_output_layout()};
		}

		m_density_network->backward(stream, *forward.density_network_ctx, forward.density_network_input, output, dL_doutput, dL_ddensity_network_input.data() ? &dL_ddensity_network_input : nullptr, use_inference_params, param_gradients_mode);

		// Backprop through pos encoding if it is trainable or if we need input gradients
		if (dL_ddensity_network_input.data()) {
			tcnn::GPUMatrixDynamic<float> dL_dpos_encoding_input;
			if (dL_dinput) {
				dL_dpos_encoding_input = dL_dinput->slice_rows(0, m_pos_encoding->input_width());
			}

			m_pos_encoding->backward(
				stream,
				*forward.pos_encoding_ctx,
				input.slice_rows(0, m_pos_encoding->input_width()),
				forward.density_network_input,
				dL_ddensity_network_input,
				dL_dinput ? &dL_dpos_encoding_input : nullptr,
				use_inference_params,
				param_gradients_mode
			);
		}
	}

	void set_params_impl(T* params, T* inference_params, T* gradients) override {
		size_t offset = 0;
		m_density_network->set_params(params + offset, inference_params + offset, gradients + offset);
		offset += m_density_network->n_params();

		m_rgb_network->set_params(params + offset, inference_params + offset, gradients + offset);
		offset += m_rgb_network->n_params();

		m_pos_encoding->set_params(params + offset, inference_params + offset, gradients + offset);
		offset += m_pos_encoding->n_params();

		m_dir_encoding->set_params(params + offset, inference_params + offset, gradients + offset);
		offset += m_dir_encoding->n_params();
	}

	void initialize_params(tcnn::pcg32& rnd, float* params_full_precision, float scale = 1) override {
		m_density_network->initialize_params(rnd, params_full_precision, scale);
		params_full_precision += m_density_network->n_params();

		m_rgb_network->initialize_params(rnd, params_full_precision, scale);
		params_full_precision += m_rgb_network->n_params();

		m_pos_encoding->initialize_params(rnd, params_full_precision, scale);
		params_full_precision += m_pos_encoding->n_params();

		m_dir_encoding->initialize_params(rnd, params_full_precision, scale);
		params_full_precision += m_dir_encoding->n_params();
	}

	size_t n_params() const override {
		return m_pos_encoding->n_params() + m_density_network->n_params() + m_dir_encoding->n_params() + m_rgb_network->n_params();
	}

	uint32_t padded_output_width() const override {
		return std::max(m_rgb_network->padded_output_width(), (uint32_t)4);
	}

	uint32_t input_width() const override {
		return m_dir_offset + m_n_dir_dims + m_n_extra_dims;
	}

	uint32_t output_width() const override {
		return 4;
	}

	uint32_t n_extra_dims() const {
		return m_n_extra_dims;
	}

	uint32_t required_input_alignment() const override {
		return 1; // No alignment required due to encoding
	}

	std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override {
		auto layers = m_density_network->layer_sizes();
		auto rgb_layers = m_rgb_network->layer_sizes();
		layers.insert(layers.end(), rgb_layers.begin(), rgb_layers.end());
		return layers;
	}

	uint32_t width(uint32_t layer) const override {
		if (layer == 0) {
			return m_pos_encoding->padded_output_width();
		} else if (layer < m_density_network->num_forward_activations() + 1) {
			return m_density_network->width(layer - 1);
		} else if (layer == m_density_network->num_forward_activations() + 1) {
			return m_rgb_network_input_width;
		} else {
			return m_rgb_network->width(layer - 2 - m_density_network->num_forward_activations());
		}
	}

	uint32_t num_forward_activations() const override {
		return m_density_network->num_forward_activations() + m_rgb_network->num_forward_activations() + 2;
	}

	std::pair<const T*, tcnn::MatrixLayout> forward_activations(const tcnn::Context& ctx, uint32_t layer) const override {
		const auto& forward = dynamic_cast<const ForwardContext&>(ctx);
		if (layer == 0) {
			return {forward.density_network_input.data(), m_pos_encoding->preferred_output_layout()};
		} else if (layer < m_density_network->num_forward_activations() + 1) {
			return m_density_network->forward_activations(*forward.density_network_ctx, layer - 1);
		} else if (layer == m_density_network->num_forward_activations() + 1) {
			return {forward.rgb_network_input.data(), m_dir_encoding->preferred_output_layout()};
		} else {
			return m_rgb_network->forward_activations(*forward.rgb_network_ctx, layer - 2 - m_density_network->num_forward_activations());
		}
	}

	const std::shared_ptr<tcnn::Encoding<T>>& pos_encoding() const {
		return m_pos_encoding;
	}

	const std::shared_ptr<tcnn::Encoding<T>>& dir_encoding() const {
		return m_dir_encoding;
	}

	const std::shared_ptr<tcnn::Network<T>>& density_network() const {
		return m_density_network;
	}

	const std::shared_ptr<tcnn::Network<T>>& rgb_network() const {
		return m_rgb_network;
	}

	tcnn::json hyperparams() const override {
		json density_network_hyperparams = m_density_network->hyperparams();
		density_network_hyperparams["n_output_dims"] = m_density_network->padded_output_width();
		return {
			{"otype", "NerfNetwork"},
			{"pos_encoding", m_pos_encoding->hyperparams()},
			{"dir_encoding", m_dir_encoding->hyperparams()},
			{"density_network", density_network_hyperparams},
			{"rgb_network", m_rgb_network->hyperparams()},
		};
	}
private:
	std::shared_ptr<tcnn::Network<T>> m_density_network;
	std::shared_ptr<tcnn::Network<T>> m_rgb_network;
	std::shared_ptr<tcnn::Encoding<T>> m_pos_encoding;
	std::shared_ptr<tcnn::Encoding<T>> m_dir_encoding;

	uint32_t m_rgb_network_input_width;
	uint32_t m_n_pos_dims;
	uint32_t m_n_dir_dims;
	uint32_t m_n_extra_dims; // extra dimensions are assumed to be part of a compound encoding with dir_dims
	uint32_t m_dir_offset;

	// // Storage of forward pass data
	struct ForwardContext : public tcnn::Context {
		tcnn::GPUMatrixDynamic<T> density_network_input;
		tcnn::GPUMatrixDynamic<T> density_network_output;
		tcnn::GPUMatrixDynamic<T> rgb_network_input;
		tcnn::GPUMatrix<T> rgb_network_output;

		std::unique_ptr<Context> pos_encoding_ctx;
		std::unique_ptr<Context> dir_encoding_ctx;

		std::unique_ptr<Context> density_network_ctx;
		std::unique_ptr<Context> rgb_network_ctx;
	};
};

NGP_NAMESPACE_END
