using Images 
using Plots
using LinearAlgebra

include("structs.jl")

function initialise_domain(image_name)
    binary_domain = Int.(Gray.(load(image_name)) .< 0.5) 
    colour_gradient = cgrad([:white, :black])
    display(
        heatmap(
            binary_domain, 
            yflip=true, 
            color=colour_gradient, 
            c=:blues, 
            legend=false, 
            title="Domain",
            xlabel="Horizontal Distance, x",
            ylabel="Vertical Distance, z"
        )
    )
    (z_dims,x_dims) = size(binary_domain)
    boundary_normals = zeros(Float64,z_dims,x_dims,2)
    for i = 2:x_dims-1, j = 2:z_dims-1
        grad_z = 0; grad_x = 0
        for ii = (i-1):(i+1)
            grad_z += binary_domain[(j-1),ii] - binary_domain[(j+1),ii]
        end
        for jj = (j-1):(j+1)
            grad_x += binary_domain[jj,(i+1)] - binary_domain[jj,(i-1)]
        end
        normals = [-grad_x/3, grad_z/3]
        if normals != [0, 0]
            boundary_normals[j,i,1] = normalize(normals)[1];
            boundary_normals[j,i,2] = normalize(normals)[2];
        end
    end
    boundary_normals[[1,end],:,:] .= boundary_normals[[2,end-1],:,:]
    boundary_normals[:,[1,end],:] .= boundary_normals[:,[2,end-1],:]
    return Domain(z_dims,x_dims,binary_domain,boundary_normals)
end

function initialise_fields(domain::Domain,vel::Velocity_Gradients)
    nx = domain.x_dims; nz = domain.z_dims
    velocity_field = zeros(nz,nx)

    for i = 1:nz, j = 1:nx
        velocity_field[i,j] = vel.base_velocity + i*vel.z_grad + j*vel.x_grad
    end

    slowness_field = 1 ./ velocity_field
    du_dx = zeros(nz,nx)
    du_dz = zeros(nz,nx)

    for i = 1:nx-1
        du_dx[:,i] = (slowness_field[:,i+1] .- slowness_field[:,i]) 
    end
    for j = 1:nz-1
        du_dz[j,:] = (slowness_field[j+1,:] .- slowness_field[j,:])
    end
    # Boundary conditions (Dirichlet)
    du_dz[[end],:] = du_dz[[end-1],:]
    du_dx[:,[end]] = du_dx[:,[end-1]]

    velocity_visualisation = copy(velocity_field)
    indices = findall(domain.binary_domain .== 1)
    [velocity_visualisation[indice] = 0 for indice in indices]

    velocity_fig = heatmap(
        velocity_visualisation,yflip=true,legend=true,
        color=:viridis, title="Velocity Field",xlabel="Horizontal Distance",
        ylabel="Vertical Distance"
    )
    display(velocity_fig)
    return Domain_Fields(velocity_field,slowness_field,du_dx,du_dz)
end

function initialise_ray(ICs::Initial_Conditions)
    ds = ICs.ds
    steps = Int(ICs.s_max / ICs.ds)
    amplitude = zeros(Float64, steps)
    amplitude[1] = ICs.amp_init
    z_pos = zeros(Float64, steps)
    z_pos[1] = ICs.z_init
    x_pos = zeros(Float64, steps)
    x_pos[1] = ICs.x_init
    time = zeros(Float64, steps)
    init_angle = ICs.theta_init
    exit_angle = []
    reflections = []
    return Ray(amplitude,z_pos,x_pos,time,ds,steps,init_angle,exit_angle,reflections)
end

function trace_ray(domain::Domain,fields::Domain_Fields,ICs)
    function get_submatrix(matrix,z,x)
        # Gets 3x3 matrix around points z,x  
        zz = Int(round(z)); xx = Int(round(x))
        if (xx in [1,size(matrix)[2]]) || (zz in [1,size(matrix)[1]])
            submatrix = matrix[zz,xx]
        else
            submatrix = matrix[(zz-1):(zz+1),(xx-1):(xx+1)]
        end
        return submatrix,zz,xx
    end
    function interpolate_field(field,x,z)
        # name ^
        x1 = Int(floor(x)); x2 = Int(ceil(x))
        z1 = Int(floor(z)); z2 = Int(ceil(z))
        if x1 == x || x2 == x 
            interp_col = field[:,Int(x)] 
        else
            col1 = field[:,x1]; col2 = field[:,x2]
            interp_col = col1 .+ (col2 .- col1).*(x-x1)./(x2-x1)
        end
        if z1 == z || z2 == z
            interp_val = interp_col[Int(z)]
        else
            row1 = interp_col[z1]; row2 = interp_col[z2]
            interp_val = row1 + (row2-row1)*(z-z1)/(z2-z1)
        end 
        if isnan(interp_val)
            interp_val = 0
        end

        return interp_val
    end

    ray = initialise_ray(ICs)
    u = interpolate_field(fields.slowness_field,ICs.x_init,ICs.z_init)
    sx = u*sind(ICs.theta_init); sz = u*cosd(ICs.theta_init)
    reflections = 0

    for i = 1:ray.steps - 1
        dx = (sx/u)*ICs.ds; dz = (sz/u)*ICs.ds; dt = u*ICs.ds
        ray.x_pos[i+1] = ray.x_pos[i] + dx
        ray.z_pos[i+1] = ray.z_pos[i] + dz 
        ray.time[i+1] = ray.time[i] + dt 
        z = ray.z_pos[i+1]; x = ray.x_pos[i+1]

        condition1 = (1 <= z <= domain.z_dims)
        condition2 = (1 <= x <= domain.x_dims)
        if condition1 && condition2
            # Ray is inside the domain 
            submatrix,node_z,node_x = get_submatrix(domain.binary_domain,z,x) 
            if sum(submatrix) > 2
                # Ray is approaching boundary 
                z1 = ray.z_pos[i]; x1 = ray.x_pos[i]
                v1 = [x-x1, z-z1]
                n = domain.boundary_normals[node_z,node_x,:]
                if n != [0, 0]
                    # Ray has not entered boundary
                    v2 = v1 - 2*dot(v1,n)*n   
                    ray.x_pos[i+1] = x1 + v2[1]
                    ray.z_pos[i+1] = z1 + v2[2]
                    u = interpolate_field(
                        fields.slowness_field,
                        ray.x_pos[i+1],
                        ray.z_pos[i+1]
                    )
                    sx = (v2[1]/ds)*u
                    sz = (v2[2]/ds)*u
                    reflections += 1
                else
                    # exited due to penetrating boundary. error
                    ray.x_pos[i+1:end] .= ray.x_pos[i];
                    ray.z_pos[i+1:end] .= ray.z_pos[i];
                    ray.amplitude[i+1:end] .= ray.amplitude[i];
                    ray.time[i+1:end] .= ray.time[i];
                    push!(ray.reflections, reflections)
                    break
                end
            else
                u = interpolate_field(fields.slowness_field,x,z)
                dudz = interpolate_field(fields.du_dz,x,z)
                dudx = interpolate_field(fields.du_dx,x,z)
                sx += dudx*ds; sz += dudz*ds 
            end
            ray_length = (i+1)*ds 
            ray.amplitude[i+1] = ICs.amp_init*sqrt(ds/ray_length)
        else
            # Ray is outside the domain 
            incoming_vector = normalize([x,z]-[ray.x_pos[i],ray.z_pos[i]])
            exit_angle = acosd(dot(incoming_vector,[0,1]))*sign(incoming_vector[1])
            ray.x_pos[i+1:end] .= ray.x_pos[i];
            ray.z_pos[i+1:end] .= ray.z_pos[i];
            ray.amplitude[i+1:end] .= ray.amplitude[i];
            ray.time[i+1:end] .= ray.time[i];
            push!(ray.exit_angle, exit_angle)
            push!(ray.reflections, reflections)
            break
        end
    end
    return ray
end

function plot_ray(domain::Domain,rays)
    colour_gradient = cgrad([:white, :black])
    ray_plot = heatmap(
        domain.binary_domain, 
        yflip=true, 
        color=colour_gradient, 
        c=:blues, 
        legend=false, 
        title="Domain",
        xlabel="Horizontal Distance, x",
        ylabel="Vertical Distance, z"
    ) 
    for ray in rays 
        ray_plot = plot!(ray.x_pos,ray.z_pos,yflip=true)
    end
    ray_plot = scatter!(
        [rays[1].x_pos[1]],
        [rays[1].z_pos[1]],
        color=:red,
        markersize=2.5
    )
    display(ray_plot)
end
