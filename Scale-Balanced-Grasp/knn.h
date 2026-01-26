#pragma once
#include "cpu/vision.h"
#include <torch/extension.h>

#ifdef WITH_CUDA
#include "cuda/vision.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>

// --- CHANGE 1: DEPRECATED HEADERS REMOVED ---
// #include <THC/THC.h>  <-- Removed: THC is legacy and no longer exists in modern PyTorch.
// extern THCState *state; <-- Removed: Global state management is now handled internally by ATen/c10.
#endif

int knn(at::Tensor& ref, at::Tensor& query, at::Tensor& idx)
{
    long batch, ref_nb, query_nb, dim, k;
    batch = ref.size(0);
    dim = ref.size(1);
    k = idx.size(1);
    ref_nb = ref.size(2);
    query_nb = query.size(2);

    float *ref_dev = ref.data_ptr<float>();
    float *query_dev = query.data_ptr<float>();
    long *idx_dev = idx.data_ptr<long>();

  if (ref.is_cuda()) {
#ifdef WITH_CUDA
    // --- CHANGE 2: MEMORY MANAGEMENT MODERNIZATION ---
    
    /* OLD WAY: 
       float *dist_dev;
       THCudaMalloc(state, (void**) &dist_dev, ref_nb * query_nb * sizeof(float)); 
       
       NEW WAY: 
       We create a temporary PyTorch tensor on the GPU. 
       This uses the CUDACachingAllocator automatically.
    */
    auto dist_tensor = torch::empty({ref_nb * query_nb}, ref.options());
    float *dist_dev = dist_tensor.data_ptr<float>();

    for (int b = 0; b < batch; b++)
    {
      // --- CHANGE 3: MODERN STREAM RETRIEVAL ---
      /*
         OLD WAY: THCState_getCurrentStream(state)
         NEW WAY: c10::cuda::getCurrentCUDAStream()
      */
      knn_device(ref_dev + b * dim * ref_nb, ref_nb, query_dev + b * dim * query_nb, query_nb, dim, k,
      dist_dev, idx_dev + b * k * query_nb, c10::cuda::getCurrentCUDAStream().stream());
    }
    
    // --- CHANGE 4: NO EXPLICIT FREE NEEDED ---
    /*
       OLD WAY: THCudaFree(state, dist_dev); 
       NEW WAY: Not needed. The dist_tensor destructor automatically handles memory 
                deallocation when it goes out of scope.
    */
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("error in knn: %s\n", cudaGetErrorString(err));
        // --- CHANGE 5: ERROR MACRO REPLACEMENT ---
        AT_ERROR("aborting"); // Replacement for THError()
    }
    return 1;
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }

    // CPU Fall (remains mostly unchanged but uses AT_ERROR for consistency)
    float *dist_dev = (float*)malloc(ref_nb * query_nb * sizeof(float));
    long *ind_buf = (long*)malloc(ref_nb * sizeof(long));
    for (int b = 0; b < batch; b++) {
    knn_cpu(ref_dev + b * dim * ref_nb, ref_nb, query_dev + b * dim * query_nb, query_nb, dim, k,
      dist_dev, idx_dev + b * k * query_nb, ind_buf);
    }

    free(dist_dev);
    free(ind_buf);

    return 1;
}
