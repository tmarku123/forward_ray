using Images 
using Plots
using LinearAlgebra

include("structs.jl")

function initialise_domain(image_name;plot=false)
    grey_img = Gray.(load(image_name))
    min_value = minimum(grey_img)
    max_value = maximum(grey_img)
    threshold = 0.7*(max_value-min_value) + min_value
    binary_domain = Int.((grey_img) .< threshold) 
    if plot == true
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
    end
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

function initialise_fields(domain::Domain,vel::VelocityGradients;plot=false)
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
    if plot == true
        velocity_fig = heatmap(
            velocity_visualisation,yflip=true,legend=true,
            color=:viridis, title="Velocity Field",xlabel="Horizontal Distance",
            ylabel="Vertical Distance"
        )
        display(velocity_fig)
    end
    return DomainFields(velocity_field,slowness_field,du_dx,du_dz)
end

function initialise_parameters(t_max,no_rays,angle_range,fields::DomainFields)
    v_max = maximum(fields.velocity_field)
    ds = 0.5
    dt = ds/v_max
    time = collect(0:dt:t_max)
    angles = 0
    if no_rays != 1
        angles = collect(range(angle_range[1],angle_range[2],no_rays))
    else
        angles = [angle_range[1]]
    end
    return Parameters(time,angles)
end

function initialise_ray_data(params::Parameters,no_samples)
    t_max = maximum(params.time)
    sampling_time = collect(range(0,t_max,no_samples));

    indices = Vector{Int64}(undef,no_samples)
    for i = 1:no_samples
        indices[i] = argmin(abs.(params.time .- sampling_time[i]))
    end

    position = zeros(Float64,no_samples,2)
    return RayData(position,indices)
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

function calculate_ray(ICs::InitialConditions,domain::Domain,
    params::Parameters,fields::DomainFields,ray_data::RayData)

    function initialise_ray(ICs::InitialConditions,params::Parameters)
        amp = ICs.amplitude
        x = ICs.position[1];    z = ICs.position[2]
    
        dt = params.time[2] - params.time[1]
        ds = dt/ICs.u
    
        sx = ICs.u*sind(ICs.theta);     sz = ICs.u*cosd(ICs.theta);
        dx = (sx/ICs.u)*ds;             dz = (sz/ICs.u)*ds; 
        x += dx;                        z += dz;
    
        position    = [x , z ]
        direction   = [dx, dz]
        slowness    = [sx, sz]
        return Ray(position,direction,amp,slowness)
    end
    function advance_ray(params::Parameters,fields::DomainFields,ray::Ray)
        x  = ray.position[1];   z  = ray.position[2]
        sx = ray.slowness[1];   sz = ray.slowness[2]
    
        u    = interpolate_field(fields.slowness_field,x,z)
        dudx = interpolate_field(fields.du_dx,x,z)
        dudz = interpolate_field(fields.du_dz,x,z)
    
        dt = params.time[2] - params.time[1]
        ds = dt/u
    
        sx += dudx*ds;                  sz += dudz*ds
        dx = (sx/u)*ds;                 dz = (sz/u)*ds; 
        x += dx;                        z += dz;
        
        position  = [x, z]
        direction = [dx, dz]
        slowness  = [sx, sz]
    
        amp = ray.amp # will do this later
        
        return Ray(position,direction,amp,slowness)
    end
    function reflect_ray(domain::Domain,params::Parameters,fields::DomainFields,ray::Ray)
        dt = params.time[2] - params.time[1]
    
        x = ray.position[1];    z = ray.position[2];
        x_node = Int(round(x)); z_node = Int(round(z));
        n = domain.boundary_normals[z_node,x_node,:];
        v1 = ray.direction
        v2 = v1 - 2*(dot(v1,n)*n)
    
        dx = v2[1]; dz = v2[2];
        x += dx;    z += dz;
        u = interpolate_field(fields.slowness_field,x,z)
        ds = (dt/u)
        sx = (dx/ds)*u; sz = (dz/ds)*u;
    
        position  = [x,z]
        direction = [dx,dz]
        slowness  = [sx,sz]
        amp = ray.amp
    
        return Ray(position,direction,amp,slowness)
    
    end
    function save_ray_data(i::Int64,ray::Ray,ray_data::RayData)
        if i in ray_data.indices
            j = findfirst(x -> x == i, ray_data.indices)
            ray_data.position[j:end,:] .= ray.position'
        end
        return ray_data
    end
    function get_submatrix(matrix,z,x)
        # Gets 3x3 matrix around points z,x  
        zz = Int(round(z)); xx = Int(round(x))
        if (xx in [1,size(matrix)[2]]) || (zz in [1,size(matrix)[1]])
            submatrix = matrix[zz,xx]
        else
            submatrix = matrix[(zz-1):(zz+1),(xx-1):(xx+1)]
        end
        return submatrix
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

    ray = initialise_ray(ICs,params)
    if size(ray_data.position,1) != 0
        ray_data.position[1,:] = ICs.position
    end
    steps = size(params.time, 1)

    for i = 2:steps-1
        x = ray.position[1];   z = ray.position[2];

        if 1 <= z <= domain.z_dims && 1 <= x <= domain.x_dims
            # Confirms ray is in domain
            submatrix = get_submatrix(domain.binary_domain,z,x)
            if sum(submatrix) > 2
                # Reflective boundary encountered 
                ray = reflect_ray(domain,params,fields,ray)
            else
                # No reflection, continue as normal
                ray = advance_ray(params,fields,ray)
            end
        else
            # Ray has left the Domain
            break
        end

        if size(ray_data.position,1) != 0
            ray_data = save_ray_data(i,ray,ray_data)
        end    

    end
    return ray, ray_data
end

function plot_ray(domain::Domain,ray_data)
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
    for data in ray_data 
        x = data.position[:,1]; z = data.position[:,2]
        ray_plot = plot!(x,z,yflip=true)
    end
    display(ray_plot)
end
