import Distributions
import NearestNeighbors

"""
CPU version. A GPU version is generate_fc_hypotheses with cuda=true

hypotheses arrays are used as combined arrays, each array hold one sample of all hypotheses
|------src hypotheses----|------dst hypotheses------|
|s0-s0...|s1..s1|s2..s2| for coalesced memory access

target_data_offset = hypotheses_count_ * sample_size
dst: sample_dim_idx * hypotheses_count_ + sample_idx + target_data_offset
src: sample_dim_idx * hypotheses_count_ + sample_idx
"""
function generate_hypotheses(
    src_feature::Matrix{Float32},
    dst_feature::Matrix{Float32},
    N_hypo::Union{Int64, Int32};
    u::NT,
    sample_dim::Int64=4,
    rand_in::Matrix{Float64}=Matrix{Float64}(undef, 0, 0)
) where NT<:Real#::Vector{Int32}

    corres_sorted::Vector{Tuple{Int64, Int64, Float64}} = match_and_sort_knn(
        src_feature, dst_feature
    )
    reverse!(corres_sorted)

    pair_idx_hypo = gen_levy_H(
        length(corres_sorted), u, sample_dim, N_hypo, rand_in=rand_in
    )
    # pair_idx_hypo = gen_uniform_H(length(corres_sorted), sample_dim, N_hypo)
    hypotheses_host::Vector{Int32} = Vector{Int32}(undef, sample_dim*N_hypo*2)
    target_data_offset = sample_dim*N_hypo
    for hypo_idx = 1:N_hypo
        for hypo_dim_idx = 1:sample_dim
            pair_idx = pair_idx_hypo[hypo_dim_idx, hypo_idx]
            src_idx, dst_idx, _ = corres_sorted[pair_idx]
            # write src hypo
            src_pos_in_hypo = (hypo_dim_idx-1) * N_hypo + hypo_idx
            dst_pos_in_hypo = src_pos_in_hypo + target_data_offset
            hypotheses_host[src_pos_in_hypo] = src_idx - 1 # convert to 0 indexed
            hypotheses_host[dst_pos_in_hypo] = dst_idx - 1 # convert to 0 indexed
        end
    end
    return hypotheses_host, pair_idx_hypo, corres_sorted
end


function match_and_sort_knn_cpu(
    source_fpfh::Matrix{Float32},
    target_fpfh::Matrix{Float32}
)::Vector{Tuple{Int64, Int64, Float64}}

    # get feature matching
    knn_tree = NearestNeighbors.KDTree(target_fpfh; leafsize = 10)
    src_n = size(source_fpfh, 2)
    target_n = size(target_fpfh, 2)

    corres = Vector{Tuple{Int64, Int64, Float64}}(undef, src_n)

    # for pt_idx in range(src_n):
    for pt_idx = 1:src_n
        nn_idxs = NearestNeighbors.inrange(
            knn_tree,
            source_fpfh[:, pt_idx],
            12,
            true
        )
        # nn_idxs, _ = NearestNeighbors.knn(
        #     knn_tree,
        #     source_fpfh[:, pt_idx],
        #     1
        # )

        if length(nn_idxs) > 0
            # print("src_id: {0}, dst_id: {1}, dist: {2}".format(pt_idx, idx[0], dist[0]))
            corres[pt_idx] = (pt_idx, nn_idxs[1], Float64(length(nn_idxs)))
        else
            # n, idx, dist = knn_tree.search_knn_vector_xd(source_fpfh.data[:, pt_idx], 1)
            nn_idxs, dist = NearestNeighbors.knn(
                knn_tree,
                source_fpfh[:, pt_idx],
                1
            )
            corres[pt_idx] = (pt_idx, nn_idxs[1], Float64(target_n))
            # println("another search with no radius")
        end
    end
    corres_sorted = sort(corres, by=x->x[3])

    return corres_sorted
end

function match_and_sort_knn_gpu(
    src_pcd::HDPcd,
    dst_pcd::HDPcd;
    dist_thres::Float64=12.0
)::Vector{Tuple{Int64, Int64, Float64}}
    target_n = dst_pcd.count
    src_n::Int64 = src_pcd.count
    # in range count
    gpu_out = CUDAFastRegistration.cuda_inrange(src_pcd, dst_pcd, dist_thres)
    inrange_counts_cpu = collect(gpu_out)
    # find nearest neighboors
    CUDAFastRegistration.cuda_NN(src_pcd, dst_pcd, out=gpu_out)
    nn_res_cpu = collect(gpu_out)


    corres = Vector{Tuple{Int64, Int64, Float64}}(undef, src_n)

    # for pt_idx in range(src_n):
    for pt_idx = 1:src_n

        inrange_count::Float64 = Float64(inrange_counts_cpu[pt_idx])

        if inrange_count > 0
            corres[pt_idx] = (pt_idx, Int64(nn_res_cpu[pt_idx]), inrange_count)
        else
            corres[pt_idx] = (pt_idx, Int64(nn_res_cpu[pt_idx]), Float64(target_n))
        end
    end
    corres_sorted = sort(corres, by=x->x[3])

    return corres_sorted
end


"""
scale::Int64=100 is the max truncation point. The random number is in between
(0, scale)
"""
function rand_levy(
    μ::NT1,
    σ::NT2,
    m::Int64,
    n::Int64;
    scale::Int64=100
)::Matrix{Float64} where {NT1<:Real, NT2<:Real}

    ld = Distributions.truncated(Distributions.Levy(μ, σ), 0, scale)
    rand_nums = rand(ld, (m, n))

    return rand_nums
end


"""
return 1 indexed index for sorted point pairs
"""
function gen_levy_H(
    count::Int64,
    u::NT,
    sample_size::Int64,
    N_trails::Int64;
    rand_in::Matrix{Float64}=Matrix{Float64}(undef, 0, 0)
)::Matrix{Int64} where NT<:Real
    
    σ = 1
    # u = -100# -1 to -10000
    @assert u <= -1 "greater than 1 doesn't make sense to have up and down pdf"
    scale = 100
    # use pre generated rand number
    rand_num_float = if length(rand_in) == 0
        tmp = rand_levy(u, σ, sample_size, N_trails, scale=scale)
        # @assert all(tmp .< scale)
        out_of_range_mask = .!(0 .< tmp .< scale)
        if sum(out_of_range_mask) > 0
            @warn "There are $(sum(out_of_range_mask)) number is
                   out of range in tmp, they are $(tmp[out_of_range_mask])"
            tmp[out_of_range_mask] .= min.(tmp[out_of_range_mask], scale-1e-8)
        end
        @assert 0 <= sum(out_of_range_mask) < 10
        tmp
    else
        @assert size(rand_in) == (sample_size, N_trails)
        rand_in
    end


    #rand_num_float .*= (Float64(count) / Float64(scale))
    hypotheses = ceil.(Int64, rand_num_float .* (Float64(count) / Float64(scale))) # TODO better use round?
    out_of_range_mask = .!(1 .<= hypotheses .<= count)
    if 0 < sum(out_of_range_mask) < 10
        @warn "There are $(sum(out_of_range_mask)) number is
               out of range, they are $(hypotheses[out_of_range_mask])"
        hypotheses[out_of_range_mask] .= min.(hypotheses[out_of_range_mask], count)
        hypotheses[out_of_range_mask] .= max.(hypotheses[out_of_range_mask], 1)
    elseif sum(out_of_range_mask) >= 10
        @assert all(1 .<= hypotheses .<= count) "all numbers have to be with in the range.
           There are $(sum(out_of_range_mask)) number is out of range,
           they are $(hypotheses[out_of_range_mask])"
    end
    return hypotheses
end


function gen_uniform_H(
    count::Int64,
    sample_size::Int64,
    N_trails::Int64
)::Matrix{Int64}
    # sample_end = length(data)
    hypotheses_uniform = rand(1:count, sample_size, N_trails)
    return hypotheses_uniform
end


# res = gen_levy_H(3353, -10, 4, 1001)
