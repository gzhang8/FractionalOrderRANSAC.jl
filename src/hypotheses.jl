using CUDA

# higher level function
function generate_fc_hypotheses(
    reg::Register,
    cdf,
    cdf_x_start::NT1,
    cdf_x_end::NT2;
    cuda::Bool=true
) where {NT1<:Real, NT2<:Real}

    src_feature, dst_feature = if reg.swapped
        reg.dst_feature, reg.src_feature
    else
        reg.src_feature, reg.dst_feature
    end
    if cuda
        corres_sorted::Vector{Tuple{Int64, Int64, Float64}} = match_and_sort_knn_gpu(reg.src_pcd, reg.dst_pcd)
        reverse!(corres_sorted)

        CUDAFastRegistration.clear_hypo(reg)

        sample_trunc_distribution_gpu(
            cdf, cdf_x_start, cdf_x_end,
            length(corres_sorted), reg.sample_size*reg.N_hypo,
            rand_data_raw=reg.rand_tmp_space_cu,
            levy_out=reg.levy_out_tmp_cu
        )

        sorted_corres_32_gpu = CUDA.CuArray{Tuple{Int32, Int32, Float32}}(corres_sorted)
        # write to hypos_out
        build_hypos_gpu(
            reg.levy_out_tmp_cu,
            sorted_corres_32_gpu,
            hypos_out=reg.hypotheses_cu
        )
        CUDA.unsafe_free!(sorted_corres_32_gpu)

    else
        hypo_h, _, _ = generate_hypotheses(
            src_feature,
            dst_feature,
            reg.N_hypo,
            u=u,
            sample_dim=reg.sample_size,
            rand_in=reg.rand_nums_cpu
        )
        # copy data to correct places
        # not need copy to cpu for now, since testing happens on GPU

        # to gpu
        # void copy_hypo_to_gpu(void* reg, int* external_hypo_h)
        ccall(
            (:copy_hypo_to_gpu, "libcuda_fast_registration"),
            Cvoid,
            (Ptr{Cvoid}, Ptr{Cint}),
            reg.c_ptr,
            Base.unsafe_convert(Ptr{Cint}, hypo_h)
        )
    end
end


# generate hypotheses from CuArray

function build_hypos_gpu(
    levy_rand_nums::CUDA.CuArray{Int32},
    crrespondences_sorted::CUDA.CuArray{Tuple{Int32, Int32, Float32}};
    hypos_out::CUDA.CuArray{Int32} = CUDA.CuArray{Int32}(undef, length(levy_rand_nums)*2)
)

    rand_size = length(levy_rand_nums)
    nthreads = 256

    if length(crrespondences_sorted) < 6000
        nblocks = Int(floor(3800 / nthreads))
        shmem_size = length(crrespondences_sorted) * sizeof(Int32) * 2
        thread_num::Int32 = nthreads * nblocks
        @cuda blocks=nblocks threads=nthreads shmem=shmem_size build_hypo_kernel(
            levy_rand_nums, crrespondences_sorted, hypos_out, thread_num
        )
    else
        nblocks = Int(ceil(rand_size/nthreads))
        @cuda blocks=nblocks threads=nthreads build_hypo_kernel_no_share(
            levy_rand_nums, crrespondences_sorted, hypos_out
        )

    end
    synchronize()

    return hypos_out
end

function build_hypo_kernel(
    levy_rand_nums::CUDA.CuDeviceVector{Int32},
    crrespondences_sorted::CUDA.CuDeviceVector{Tuple{Int32, Int32, Float32}},
    hypos_out::CUDA.CuDeviceVector{Int32},
    thread_num::Int32
)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    target_data_offset = length(levy_rand_nums)

    # load shared mem
    corres_num = length(crrespondences_sorted)
    # mems: src ....... | dst ..... |
    shmem = @cuDynamicSharedMem(Int32, corres_num*2)
    for j=0:Int32(ceil(corres_num/blockDim().x))-1
        corres_idx = j * blockDim().x + threadIdx().x
        if corres_idx <= corres_num
            src_idx, dst_idx, _ = crrespondences_sorted[corres_idx]
            shmem[corres_idx] = src_idx
            shmem[corres_idx+corres_num] = dst_idx
        end
    end
    sync_threads()

    # use loop and hope to reuse shared mem
    for loop_idx = 0:Int(ceil(length(levy_rand_nums)/thread_num))-1
        data_idx = loop_idx * thread_num + i
        if data_idx <= length(levy_rand_nums)
            pair_idx = levy_rand_nums[data_idx] #pair_idx_hypo[hypo_dim_idx, hypo_idx]
            # src_idx, dst_idx, _ = crrespondences_sorted[pair_idx]
            src_idx = shmem[pair_idx]
            dst_idx = shmem[pair_idx+corres_num]
            # write src hypo
            src_pos_in_hypo = data_idx
            dst_pos_in_hypo = src_pos_in_hypo + target_data_offset
            hypos_out[src_pos_in_hypo] = src_idx - 1 # convert to 0 indexed
            hypos_out[dst_pos_in_hypo] = dst_idx - 1 # convert to 0 indexed
        end
    end
    return

end

function build_hypo_kernel_no_share(
    levy_rand_nums::CUDA.CuDeviceVector{Int32},
    crrespondences_sorted::CUDA.CuDeviceVector{Tuple{Int32, Int32, Float32}},
    hypos_out::CUDA.CuDeviceVector{Int32}
)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    target_data_offset = length(levy_rand_nums)
    if i <= length(levy_rand_nums)
        pair_idx = levy_rand_nums[i] #pair_idx_hypo[hypo_dim_idx, hypo_idx]
        src_idx, dst_idx, _ = crrespondences_sorted[pair_idx]
        # write src hypo
        src_pos_in_hypo = i
        dst_pos_in_hypo = src_pos_in_hypo + target_data_offset
        hypos_out[src_pos_in_hypo] = src_idx - 1 # convert to 0 indexed
        hypos_out[dst_pos_in_hypo] = dst_idx - 1 # convert to 0 indexed
    end
    return

end

function build_hypos_cpu(
    levy_rand_nums::Vector{Int32},
    crrespondences_sorted::Vector{Tuple{Int32, Int32, Float32}}
)::Vector{Int32}

    rand_size = length(levy_rand_nums)
    # pair_idx_hypo = gen_uniform_H(length(corres_sorted), sample_dim, N_hypo)
    hypotheses_host::Vector{Int32} = Vector{Int32}(undef, rand_size*2)
    target_data_offset = rand_size
    for i = 1:rand_size
        # for hypo_dim_idx = 1:sample_dim
            pair_idx = levy_rand_nums[i] #pair_idx_hypo[hypo_dim_idx, hypo_idx]
            src_idx, dst_idx, _ = crrespondences_sorted[pair_idx]
            # write src hypo
            src_pos_in_hypo = i
            dst_pos_in_hypo = src_pos_in_hypo + target_data_offset
            hypotheses_host[src_pos_in_hypo] = src_idx - 1 # convert to 0 indexed
            hypotheses_host[dst_pos_in_hypo] = dst_idx - 1 # convert to 0 indexed
        # end
    end
    return hypotheses_host
end

