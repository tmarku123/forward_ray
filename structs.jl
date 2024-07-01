
struct Ray
    amplitude::Vector{Float64}
    z_pos::Vector{Float64}
    x_pos::Vector{Float64}
    time::Vector{Float64}
    ds::Float64
    steps::Int64
    init_angle::Float64
    exit_angle::Vector{Float64}
    reflections::Vector{Float64}
end

struct Initial_Conditions
    x_init::Float64
    z_init::Float64
    amp_init::Float64
    theta_init::Float64
    ds::Float64
    s_max::Int64
end

struct Domain
   z_dims::Int64
   x_dims::Int64
   binary_domain::Matrix{Bool}
   boundary_normals::Array{Float64, 3}
end

struct Domain_Fields
    velocity_field::Matrix{Float64}
    slowness_field::Matrix{Float64}
    du_dx::Matrix{Float64}
    du_dz::Matrix{Float64}
end

struct Velocity_Gradients
    x_grad::Float64
    z_grad::Float64
    base_velocity::Float64
end
