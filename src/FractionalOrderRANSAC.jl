module FractionalOrderRANSAC

using CUDAFastRegistration
import LinearAlgebra

include("fc_sampling.jl")
include("trunc_levy_int_sampling.jl")
include("hypotheses.jl")


function cu_align_fo(
    reg::Register;
    nn_feature_k::Int64,
    max_correspondence_distance::Float64,
    inlier_fraction::Float64,
    inlier_number::Int64,
    edge_similarity::Float64,
    cdf,
    cdf_x_start::NT1=0,
    cdf_x_end::NT2=100,
) where {NT1<:Real, NT2<:Real}

    CUDAFastRegistration.set_nn_feaure_k(reg, nn_feature_k)

    CUDAFastRegistration.set_max_correspondence_distance(reg, max_correspondence_distance)
    # Inlier threshold
    CUDAFastRegistration.set_min_inlier_fraction(reg, inlier_fraction)
    # Required inlier fraction for accepting a pose hypothesis
    CUDAFastRegistration.set_min_inlier_number(reg, inlier_number)

    CUDAFastRegistration.set_edge_similarity(reg, edge_similarity)

    generate_fc_hypotheses(reg, cdf, cdf_x_start, cdf_x_end)

    CUDAFastRegistration.cuPolyRejection_c(reg, edge_similarity)
    converged::Bool = CUDAFastRegistration.cudaRANSANC_c(reg)
    return converged
end


function cu_align_fo(
    reg::Register,
    src::HDPcd,
    dst::HDPcd;
    nn_feature_k::Int64,
    max_correspondence_distance::Float64,
    inlier_fraction::Float64,
    inlier_number::Int64,
    edge_similarity::Float64,
    smart_swap::Bool = false,
    cdf,
    cdf_x_start::NT1=0,
    cdf_x_end::NT2=100,
) where {NT1<:Real, NT2<:Real}
    swapped = false
    if smart_swap
        if src.count > dst.count
            swapped = true
            src, dst = dst, src
        end
    end
    reg.swapped = swapped
    CUDAFastRegistration.set_src_pcd(reg, src)
    CUDAFastRegistration.set_target_pcd(reg, dst)
    converged = cu_align_fo(
        reg,
        nn_feature_k = nn_feature_k,
        max_correspondence_distance = max_correspondence_distance,
        inlier_fraction = inlier_fraction,
        inlier_number = inlier_number,
        edge_similarity = edge_similarity,
        cdf=cdf,
        cdf_x_start=cdf_x_start,
        cdf_x_end=cdf_x_end,
    )
    T = Matrix{Float64}(LinearAlgebra.I, 4, 4)
    if converged
        T = CUDAFastRegistration.get_final_transformation(reg)
        if swapped
            T = inv(T)
        end
    end
    return T, converged
end

end # module
