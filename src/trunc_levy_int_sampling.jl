import Distributions
import Random

using CUDA

## gpu version

# rand to rand_levy map kernel with shared mem
function uni2cdf_map_kernel(
    uniform_rand_nums::CUDA.CuDeviceVector{Float64},
    cdf_steps::CUDA.CuDeviceVector{Float64},
    levy_out::CUDA.CuDeviceVector{Int32},
    N_idx::Int32
)
    idx = (blockIdx().x-1) * blockDim().x + threadIdx().x

    shmem = @cuDynamicSharedMem(Float64, length(cdf_steps))
    # load data to shared mem
    for j=1:Int32(ceil(length(cdf_steps)/blockDim().x))
        data_idx = (j-1) * blockDim().x + threadIdx().x
        if data_idx <= length(cdf_steps)
            shmem[data_idx] = cdf_steps[data_idx]
        end
    end

    sync_threads()
    # do converting

    if idx <= length(uniform_rand_nums)

        num = uniform_rand_nums[idx]

        # check first bin
        if num < shmem[1]
            levy_out[idx] = 1
        end

        # # check last bin TODO first last check one as special should be enough
        # if num > all_cdf1_scale[N_idx-1]
        #     int_rand_res[idx] = N_idx
        # end

        # check middle ones
        low_idx::Int32 = 2
        high_idx::Int32 = N_idx #- 1
        # out_idx::Int32 = -1
        # current idx is up bound(exclude), -1 is low (includ) bound
        while low_idx <= high_idx
            mid_idx = Int32(floor((low_idx+high_idx)/2))
            if @inbounds shmem[mid_idx-1] <= num < @inbounds shmem[mid_idx]
                # good condition
                @inbounds levy_out[idx] = mid_idx
                break
            elseif @inbounds shmem[mid_idx-1] > num
                # even low bound is greater
                high_idx = mid_idx-1
                continue
            elseif @inbounds shmem[mid_idx] <= num
                # high bound is lower than current
                low_idx = mid_idx + 1
                continue
            end
        end

    end

    return
end

# rand to rand_levy map kernel
function uni2cdf_map_kernel_no_share(
    uniform_rand_nums::CUDA.CuDeviceVector{Float64},
    cdf_steps::CUDA.CuDeviceVector{Float64},
    levy_out::CUDA.CuDeviceVector{Int32},
    N_idx::Int32
)
    idx = (blockIdx().x-1) * blockDim().x + threadIdx().x


    if idx <= length(uniform_rand_nums)
        num = uniform_rand_nums[idx]

        # check first bin
        if num < cdf_steps[1]
            levy_out[idx] = 1
        end

        # # check last bin TODO first last check one as special should be enough
        # if num > all_cdf1_scale[N_idx-1]
        #     int_rand_res[idx] = N_idx
        # end

        # check middle ones
        low_idx = 2
        high_idx = N_idx #- 1
        # current idx is up bound(exclude), -1 is low (includ) bound
        while low_idx <= high_idx
            mid_idx = Int(floor((low_idx+high_idx)/2))
            if cdf_steps[mid_idx-1] <= num < cdf_steps[mid_idx]
                # good condition
                levy_out[idx] = mid_idx
                break
            elseif cdf_steps[mid_idx-1] > num
                # even low bound is greater
                high_idx = mid_idx-1
                continue
            elseif cdf_steps[mid_idx] <= num
                # high bound is lower than current
                low_idx = mid_idx + 1
                continue
            end

        end
    end


    return
end

"""
this function can run with any cdf
    cdf is a cdf function for a Distributions

the final sampled number are within [1, max_idx]
n controls the number of numbers to be sampled.
scale::Int64=100, @deprecate

Output is a CuArrays
"""
function sample_trunc_distribution_gpu(
    cdf,
    cdf_x_start::NT1,
    cdf_x_end::NT2,
    max_idx::Union{Int64, Int32},
    n::Int64;
    rand_data_raw=CUDA.CuArray{Float64}(undef, n),
    levy_out=CUDA.CuArray{Int32}(undef, n)
) where {NT1<:Real, NT2<:Real}

    lcdf = cdf(cdf_x_start)
    hcdf = cdf(cdf_x_end)

    tp = hcdf - lcdf # use this to scale the rand (0, 1)
    # step 2 participate scale in range to N numbers
    # [] at the end will remove the first item which is cdf_x_start
    distributed_scale = LinRange(cdf_x_start, cdf_x_end, max_idx+1)[end-max_idx+1:end]

    # step 3 look back on on the x value for the N cdf
    all_cdf = cdf.(distributed_scale)
    all_cdf .= (all_cdf .- lcdf) ./ tp

    # step 4 rand uniform and then check with the N x value, within the range,
    #        then its that interger

    Random.rand!(rand_data_raw)

    # can find the bucket idx with binary search
    all_cdf_cu = CUDA.adapt(CUDA.CuArray{Float64}, all_cdf)


    # caculate block numbers and thread numbers
    nthreads = 128
    nblocks = Int(ceil(length(rand_data_raw) / nthreads))
    if max_idx < 6000 # check for shmem
        shmem_size = max_idx * sizeof(Float64)
        @cuda blocks=nblocks threads=nthreads shmem=shmem_size uni2cdf_map_kernel(
            rand_data_raw, all_cdf_cu, levy_out, Int32(max_idx)
        )
    else
        # non shmem version
        @cuda blocks=nblocks threads=nthreads uni2cdf_map_kernel_no_share(
            rand_data_raw, all_cdf_cu, levy_out, Int32(max_idx)
        )

    end
    synchronize()
    CUDA.unsafe_free!(all_cdf_cu)
    return levy_out
end


# cpu version
"""
max_idx is the index you want to generate: 1 to max_idx
[0, scale] is the portion we use to generate numbers
"""
function sample_trunc_distribution_cpu(
    cdf,
    cdf_x_start::NT1,
    cdf_x_end::NT2,
    max_idx::Union{Int64, Int32},
    n::Int64
)::Vector{Int32} where {NT1<:Real, NT2<:Real}
    # @assert false "we do not use cpu version for now"
    # ld = Distributions.Levy(μ, σ)

    lcdf = cdf(cdf_x_start)
    hcdf = cdf(cdf_x_end)
    tp = hcdf - lcdf # use this to scale the rand (0, 1)
    # step 2 participate scale in range to N numbers
    distributed_scale = LinRange(cdf_x_start, cdf_x_end, max_idx+1)[end-max_idx+1:end]

    # step 3 look back on on the x value for the N cdf
    all_cdf = cdf.(distributed_scale)
    all_cdf .= (all_cdf .- lcdf) ./ tp

    # step 4 rand uniform and then check with the N x value, within the range,
    #        then its that interger
    rand_data_raw3 = CUDA.CuArray{Float64}(undef, n);
    Random.rand!(rand_data_raw3)
    rand_data_raw3_cpu = CUDA.collect(rand_data_raw3)

    # can find the bucket idx with binary search
    int_rand_res = Vector{Int32}(undef, length(rand_data_raw3))
    int_rand_res .= -1
    for (idx, num) = enumerate(rand_data_raw3_cpu)

        # check first bin
        if num < all_cdf[1]
            int_rand_res[idx] = 1
        end

        # check middle ones
        low_idx = 2
        high_idx = max_idx #- 1
        # current idx is up bound(exclude), -1 is low (includ) bound
        while low_idx <= high_idx
            mid_idx = Int(floor((low_idx+high_idx)/2))
            if all_cdf[mid_idx-1] <= num < all_cdf[mid_idx]
                # good condition
                int_rand_res[idx] = mid_idx
                break
            elseif all_cdf[mid_idx-1] > num
                # even low bound is greater
                high_idx = mid_idx-1
                continue
            elseif all_cdf[mid_idx] <= num #TODO check =
                # high bound is lower than current
                low_idx = mid_idx + 1
                continue
            end

        end
        @assert int_rand_res[idx] != -1 "low_idx: $low_idx, high_idx:$high_idx,
            num: $num, low_value: $(all_cdf[low_idx]),
            high_value: $(all_cdf[high_idx])
        "

    end

    @assert sum(int_rand_res .== -1) == 0

    return int_rand_res

end


## test

# σ = 1
# u_array = [-1, -10, -100, -1000, -1e4, -1e5, -1e6]
# u = u_array[2]
#
# cpu_res = sample_trunc_levy_cpu(u, σ, 3000, 4*4000000)
#
# gpu_res = sample_trunc_levy_gpu(u, σ, 3000, 4*4000000)
# #     μ::NT1,
# #     σ::NT2,
# #     max_idx::Union{Int64, Int32},
# #     n::Int64;
# #     scale::Int64=100
# # )
# n = 4*4000000
# rand_data_raw = CuArrays.CuArray{Float64}(undef, n)
# levy_out=CuArrays.CuArray{Int32}(undef, n)
# sample_trunc_levy_gpu(u, σ, 3000, 4*4000000, rand_data_raw=rand_data_raw, levy_out=levy_out)
